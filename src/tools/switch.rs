use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::analysis::{self, Analyzer, Control, FuncAnalysis};
use crate::exec_state::{ExecutionState, VirtualAddress};
use crate::{BinaryFile, Operation, Rva, OperandCtx, Operand};

pub struct FoundSwitch<'e, Va: VirtualAddress> {
    branch_start: Va,
    switch: SwitchParts<'e, Va>,
    first_case: u32,
    case_count: u32,
}

pub struct SwitchParts<'e, Va: VirtualAddress> {
    base: Va,
    main_index: Operand<'e>,
    u32_cases: Va,
    u8_cases: Option<Va>,
}

/// Should be called on `Operation::Jump` when jumping to a non-constant operand.
/// `to` must be resolved.
/// Caller should do something to store the result later for `handle_found_switches`,
/// and possibly avoid calling this function again if the branch is reached again.
pub fn handle_switch<'e, A, E, Va>(
    ctrl: &mut Control<'e, '_, '_, A>,
    to: Operand<'e>,
) -> Option<FoundSwitch<'e, Va>>
where E: ExecutionState<'e, VirtualAddress = Va>,
      Va: VirtualAddress,
      A: Analyzer<'e, Exec = E>,
{
    let binary = ctrl.binary();
    let ctx = ctrl.ctx();
    if let Some(switch) = extract_switch_parts(to, ctx, binary.base()) {
        // Add comments for each branch
        let main_index = switch.main_index;
        let state = ctrl.exec_state();
        let limits = state.value_limits(main_index);
        let case_count = if limits.1 >= limits.0 {
            if limits.1 - limits.0 < 0x1000 {
                (limits.1 - limits.0 + 1) as u32
            } else {
                0x1000
            }
        } else {
            0
        };
        let found_switch = FoundSwitch {
            branch_start: ctrl.branch_start(),
            switch,
            first_case: limits.0 as u32,
            case_count,
        };
        Some(found_switch)
    } else {
        // Not switch
        None
    }
}

fn extract_switch_parts<'e, Va: VirtualAddress>(
    to: Operand<'e>,
    ctx: OperandCtx<'e>,
    binary_base: Va,
) -> Option<SwitchParts<'e, Va>> {
    // Scarf has a better check, but this should be fine here
    // TODO too specific for x86-64 msvc.
    let (offset, r) = to.if_arithmetic_add()?;
    let base = r.if_constant()?;
    let offset_mem = offset.if_mem32()?;
    if base != binary_base.as_u64() {
        return None;
    }

    let (u32_index, cases_u32) = offset_mem.address();
    let u32_index = u32_index
        .if_arithmetic_mul()
        .filter(|(_, r)| r.if_constant() == Some(4))
        .map(|(l, _)| l)
        .unwrap_or_else(|| ctx.rsh_const(u32_index, 2));
    let index_cases_u8 = u32_index.if_mem8()
        .and_then(|mem| {
            let (index, base) = mem.address();
            let base = Va::from_u64(base);
            if base < binary_base {
                None
            } else {
                Some((index, base))
            }
        });
    let main_index = match index_cases_u8 {
        Some((u8_index, _)) => u8_index,
        None => u32_index,
    };
    Some(SwitchParts {
        base: Va::from_u64(base),
        main_index,
        u8_cases: index_cases_u8.map(|x| x.1),
        u32_cases: Va::from_u64(cases_u32),
    })
}

#[derive(Clone, Debug)]
pub struct SwitchInfo<'e, Va: VirtualAddress> {
    /// The variable which is being switched on.
    pub main_index: Operand<'e>,
    /// Lowest value that the switch has cases for.
    /// May be determined from a branch preceding the switch.
    /// If `cases` is empty this value is not meaningful.
    pub first_case: u32,
    /// Highest value that the switch has cases for (inclusive).
    /// May be determined from a branch preceding the switch.
    /// If `cases` is empty this value is not meaningful.
    pub last_case: u32,
    /// Reverse map of case address -> index value.
    /// Maybe should be u64 -> Va map instead? And have user convert to this
    /// if they want to know if multiple cases map to same address.
    pub cases: BTreeMap<Va, Vec<u64>>,
}

/// Converts `FoundSwitch` to `SwitchInfo`, intended to be called at analysis end
/// as this may need to use the full CFG.
/// `SwitchInfo` will contain mapping of destination -> case indices.
pub fn handle_found_switch<'a, 'e, 'l, E, Va, S, F>(
    switch: &FoundSwitch<'e, Va>,
    ctx: OperandCtx<'e>,
    binary: &'e BinaryFile<Va>,
    cfg: &mut analysis::Cfg<'e, E, S>,
) -> SwitchInfo<'e, Va>
where Va: VirtualAddress,
      E: ExecutionState<'e, VirtualAddress = Va>,
      S: analysis::AnalysisState,
      F: FnMut(Va, String),
{
    let first_case = switch.first_case;
    let case_count = switch.case_count;
    let branch_start = switch.branch_start;
    let switch = &switch.switch;
    let main_index = switch.main_index;
    let mut case_offset = 0u32;
    if first_case == 0 && unwrap_sext(main_index).is_undefined() {
        // If two different branches merge to a branch doing switch jump, e.g.
        // sub eax, 6
        // cmp eax, 6
        // ja no_switch
        // (calculate switch dest with index = eax, first case = 6)
        // jmp (switch dest)
        //
        // The switch branch may see index = undefined, first case = 0
        // due to branches being merged.
        // Check switch branch predecessors, if there is such jump and register which
        // got checked gets used as switch index, apply offset to these comments
        let offset = switch_preceding_branch_offset(ctx, binary, cfg, branch_start);
        if let Some(offset) = offset {
            case_offset = offset;
        }
    }
    let mut i = 0u32;
    let text = binary.section(b".text\0\0\0").unwrap();
    let mut cases: BTreeMap<Va, Vec<u64>> = BTreeMap::new();
    while i < case_count {
        let idx = match first_case.checked_add(i) {
            Some(o) => o,
            None => break,
        };
        let u32_index = match switch.u8_cases {
            Some(cases_u8) => {
                binary.read_u8(cases_u8 + idx).ok().map(|x| x as u32)
            }
            None => Some(idx),
        };
        let dest = u32_index
            .and_then(|idx| binary.read_u32(switch.u32_cases + idx * 4).ok())
            .map(|uint32| switch.base + uint32)
            .filter(|&dest| dest >= text.virtual_address)
            .filter(|&dest| dest < text.virtual_address + text.virtual_size);
        let dest = match dest {
            Some(s) => s,
            None => break,
        };
        cases.entry(dest).or_default().push(idx as u64);
        i += 1;
    }
    let offset_first_case = first_case.saturating_add(case_offset);
    let switch_info = SwitchInfo {
        main_index,
        first_case: offset_first_case,
        last_case: offset_first_case + (i - 1),
        cases,
    };
    switch_info
}

fn unwrap_sext<'e>(op: Operand<'e>) -> Operand<'e> {
    match *op.ty() {
        crate::operand::OperandType::SignExtend(val, ..) => val,
        _ => op,
    }
}

fn switch_preceding_branch_offset<'e, E, Va, S>(
    ctx: OperandCtx<'e>,
    binary: &'e BinaryFile<Va>,
    cfg: &mut analysis::Cfg<'e, E, S>,
    switch_branch: Va,
) -> Option<u32>
where Va: VirtualAddress,
      E: ExecutionState<'e, VirtualAddress = Va>,
      S: analysis::AnalysisState,
{
    let predecessors = cfg.predecessors();
    let mut results = SwitchBranchResult::None;
    let base = binary.base();
    let rva = Rva((switch_branch.as_u64() - base.as_u64()) as u32);
    let switch_node = match cfg.get_link(rva) {
        Some(o) => o,
        None => {
            error!("No switch branch at {:?} ???", switch_branch);
            return None;
        }
    };

    for branch in predecessors.predecessors(cfg, &switch_node) {
        let branch_addr = base + branch.address().0;
        let mut analysis = FuncAnalysis::new(binary, ctx, branch_addr);
        let mut analyzer = GetJumpOffset::<E> {
            result: None,
            phantom: Default::default(),
        };
        analysis.analyze(&mut analyzer);
        results = match (results, analyzer.result) {
            (_, None) => SwitchBranchResult::Many,
            (SwitchBranchResult::None, Some(x)) => SwitchBranchResult::One(x),
            (SwitchBranchResult::One(x), Some(y)) if x == y => SwitchBranchResult::One(x),
            (_, Some(_)) => SwitchBranchResult::Many,
        };
    }
    match results {
        SwitchBranchResult::None | SwitchBranchResult::Many => None,
        SwitchBranchResult::One(val) => Some(val),
    }
}

struct GetJumpOffset<'e, E: ExecutionState<'e>> {
    result: Option<u32>,
    phantom: PhantomData<(&'e (), *mut E)>,
}

impl<'e, E: ExecutionState<'e>> analysis::Analyzer<'e> for GetJumpOffset<'e, E> {
    type State = analysis::DefaultState;
    type Exec = E;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        if let Operation::Jump { condition, .. } = *op {
            let condition = ctrl.resolve(condition);
            if let Some((l, r)) = condition.if_arithmetic_gt() {
                if r.if_constant().is_some() {
                    if let Some((_, r)) = l.if_arithmetic_sub() {
                        if let Some(c) = r.if_constant() {
                            self.result = u32::try_from(c).ok();
                        }
                    }
                }
            }
            ctrl.end_analysis();
        }
    }

    fn branch_end(&mut self, ctrl: &mut Control<'e, '_, '_, Self>) {
        ctrl.end_analysis();
    }
}

enum SwitchBranchResult {
    None,
    One(u32),
    Many,
}

use scarf::analysis::{self, Control, DefaultState};
use scarf::{BinarySection, BinaryFile, OperandContext, OperandCtx, Operation, VirtualAddress64};

#[test]
fn continue_at_1() {
    // Test that continue_at jne works
    let mut a = ContinueAt1(false);
    let bin = make_bin(&[
        0xc7, 0x00, 0x00, 0x45, 0x00, 0x00, // mov [eax], 4500
        0x50, // push eax
        0xf6, 0x00, 0x45, // test byte [eax], 45
        0x75, 0x01, // jne end
        0xcc,
        0xc3, // ret
    ]);
    let ctx = &OperandContext::new();
    test_inline(&bin, ctx, &mut a);
    assert!(a.0);
}

struct ContinueAt1(bool);
impl<'e> analysis::Analyzer<'e> for ContinueAt1 {
    type State = analysis::DefaultState;
    type Exec = scarf::ExecutionStateX86_64<'e>;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        if let Operation::Jump { to, .. } = *op {
            let to = ctrl.resolve(to).if_constant().unwrap();
            ctrl.continue_at_address(VirtualAddress64(to));
        }
        if let Operation::Return(..) = *op {
            self.0 = true;
        }
        if let Operation::Error(..) = *op {
            panic!("Reached int3");
        }
    }
}

#[test]
fn continue_at_2() {
    // Test that continue_at for the first operation of push
    // still executes the second too.
    let mut a = ContinueAt2(0);
    let bin = make_bin(&[
        0xc7, 0x00, 0x00, 0x45, 0x00, 0x00, // mov [eax], 4500
        0x50, // push eax
        0xf6, 0x00, 0x45, // test byte [eax], 45
        0x75, 0x01, // jne end
        0xcc,
        0xc3, // ret
    ]);
    let ctx = &OperandContext::new();

    test_inline(&bin, ctx, &mut a);
    assert_eq!(a.0, 5);
}

struct ContinueAt2(u8);
impl<'e> analysis::Analyzer<'e> for ContinueAt2 {
    type State = analysis::DefaultState;
    type Exec = scarf::ExecutionStateX86_64<'e>;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        assert!(self.0 != 3 && self.0 != 5);
        if let Operation::Move(_, value) = *op {
            let value = ctrl.resolve(value);
            if value.if_constant() == Some(0x4500) {
                assert_eq!(self.0, 0);
                self.0 = 1;
            } else {
                if self.0 == 1 {
                    let cont = ctrl.address() + 7;
                    ctrl.continue_at_address(cont);
                    self.0 = 2;
                } else {
                    assert_eq!(self.0, 2);
                    self.0 = 3;
                }
            }
        }
        if let Operation::Return(..) = *op {
            let ctx = ctrl.ctx();
            // Check that push worked; rsp has been offset by -8 and value at [rsp] is rax
            let rsp = ctrl.resolve(ctx.register(4));
            let expected = ctx.sub_const(ctx.register(4), 8);
            assert_eq!(rsp, expected);

            let deref_rsp = ctrl.resolve(ctx.mem64(ctx.register(4), 0));
            let expected = ctx.register(0);
            assert_eq!(deref_rsp, expected);

            assert_eq!(self.0, 4);
            self.0 = 5;
        }
        if let Operation::Error(..) = *op {
            panic!("Reached int3");
        }
    }

    fn branch_start(&mut self, _: &mut Control<'e, '_, '_, Self>) {
        if self.0 != 0 {
            assert_eq!(self.0, 3);
            self.0 = 4;
        }
    }
}

#[test]
fn analyze_address_0() {
    // Verify that analyzing address 0 is no-op
    // Used to be for backwards compat so will keep it that way.
    // It can easily happen with memory write of 0 to variable which is later called.
    let mut a = AnalyzeAddress0(0);
    let bin = make_bin(&[
        0xc3, // ret
    ]);
    let ctx = &OperandContext::new();

    test_inline(&bin, ctx, &mut a);
    assert_eq!(a.0, 1);
}

struct AnalyzeAddress0(u8);
impl<'e> analysis::Analyzer<'e> for AnalyzeAddress0 {
    type State = analysis::DefaultState;
    type Exec = scarf::ExecutionStateX86_64<'e>;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        self.0 += 1;
        if let Operation::Return(..) = *op {
            ctrl.analyze_with_current_state(self, VirtualAddress64(0));
        }
    }
}

fn make_bin(code: &[u8]) -> BinaryFile<VirtualAddress64> {
    scarf::raw_bin(VirtualAddress64(0x00400000), vec![BinarySection {
        name: {
            // ugh
            let mut x = [0; 8];
            for (out, &val) in x.iter_mut().zip(b".text\0\0\0".iter()) {
                *out = val;
            }
            x
        },
        virtual_address: VirtualAddress64(0x401000),
        virtual_size: code.len() as u32,
        data: code.into(),
    }])
}

fn test_inline<'e, A>(
    binary: &'e BinaryFile<VirtualAddress64>,
    ctx: OperandCtx<'e>,
    analyzer: &mut A,
)
where A: analysis::Analyzer<'e, Exec = scarf::ExecutionStateX86_64<'e>, State = DefaultState>
{
    test_inner(binary, ctx, binary.code_section().virtual_address, analyzer);
}

fn test_inner<'e, A>(
    file: &'e BinaryFile<VirtualAddress64>,
    ctx: OperandCtx<'e>,
    func: VirtualAddress64,
    analyzer: &mut A,
)
where A: analysis::Analyzer<'e, Exec = scarf::ExecutionStateX86_64<'e>, State = DefaultState>
{
    let state = scarf::ExecutionStateX86_64::with_binary(file, ctx);
    let mut analysis = analysis::FuncAnalysis::with_state(file, ctx, func, state);
    analysis.analyze(analyzer);
}

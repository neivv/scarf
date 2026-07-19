extern crate clap;
extern crate scarf;

use scarf::{VirtualAddress};

fn main() {
    let matches = clap::App::new("scarf-cfg")
        .arg(clap::Arg::with_name("path")
            .index(1)
            .value_name("FILE")
            .required(true)
            .help("Selects the binary"))
        .arg(clap::Arg::with_name("addr")
            .index(2)
            .value_name("ADDRESS")
            .required(true)
            .help("Function start address (in hex)"))
        .arg(clap::Arg::with_name("destructive_calls")
            .long("destructive_calls")
            .required(false)
            .takes_value(false)
            .help("Assumes that calls may write to any pointer passed to them"))
        .get_matches();
    let file = matches.value_of_os("path").unwrap();
    // Address is with default base
    let addr = matches.value_of("addr").unwrap();
    let addr = u32::from_str_radix(&addr, 16).expect("Address wasn't hex");
    let destructive_calls = matches.is_present("destructive_calls");
    let binary = scarf::parse(file).unwrap();
    let ctx = scarf::operand::OperandContext::new();
    let analysis = scarf::analysis::FuncAnalysis::new(&binary, &ctx, VirtualAddress(addr));
    let mut was_called = false;
    let (mut cfg, errors) = analysis.finish_with_changes(|op, state, _, _| {
        if was_called {
            was_called = false;
            if destructive_calls {
                state.memory = scarf::exec_state::Memory::new();
            }
        }
        match *op {
            scarf::Operation::Call(..) => {
                was_called = true;
            }
            _ => (),
        }
    });
    for (addr, e) in errors {
        eprintln!("{:08x}: {}", addr.0, e);
    }
    scarf::cfg_dot::write(&mut cfg, &mut std::io::stdout()).unwrap();
}

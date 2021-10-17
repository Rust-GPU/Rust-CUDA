mod extract_llfns;

use pico_args::Arguments;
use std::{error::Error, path::Path};

use crate::extract_llfns::extract_llfns;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = Arguments::from_env();
    let sub = args.subcommand()?.unwrap_or_default();

    match sub.as_str() {
        "extract_llfns" => {
            let arg1 = args.free_from_str::<String>()?;
            let file = Path::new(&arg1);
            let arg2 = args.free_from_str::<String>()?;
            let dir = Path::new(&arg2);
            args.finish();
            extract_llfns(file, dir);
            Ok(())
        }
        _ => panic!("Unknown command, available: `extract_llfns`"),
    }
}

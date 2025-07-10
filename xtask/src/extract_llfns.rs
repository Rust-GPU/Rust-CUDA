//! Uses llvm-extract to extract every function from an LLVM IR file
//! and dumps them as individual standalone LLVM IR files in a directory.

use rayon::prelude::*;
use regex::Regex;
use std::{path::Path, process::Command};

// change this if you want it to print a status before running the command.
// this is useful if running the command causes segfaults.
const PRINT_EVERY_EXECUTION: bool = false;

#[allow(unused_variables)]
fn run_command_for_each_fn(func: &str, contents: &str) -> bool {
    // put any code you want to use for debugging in here. For example,
    // running nvvm on every function to see what panics.
    // `true` will keep running the command, `false` will stop running.
    //
    // !!!!! Make sure to delete any changes to this function if committing !!!!!
    true
}

// ------------------------------------------------

pub(crate) fn extract_llfns(file: &Path, dir: &Path) {
    let contents = std::fs::read_to_string(file).unwrap();
    let re = Regex::new(r#"define .*(_Z.*?)(\(|")"#).unwrap();
    let names = re
        .captures_iter(&contents)
        .map(|x| x.get(1).unwrap().as_str())
        .collect::<Vec<_>>();

    let mut contents = names
        .par_iter()
        .filter_map(|name| {
            let out_file = format!("{}/{}.ll", dir.display(), name);
            let _ = Command::new("llvm-extract")
                .arg(file)
                .arg(format!("--func={name}"))
                .arg("-S")
                .arg("--recursive")
                .arg("-o")
                .arg(&out_file)
                .spawn();

            let read = std::fs::read_to_string(format!("{}/{}.ll", dir.display(), name)).ok()?;
            Some((name, read, true))
        })
        .collect::<Vec<_>>();

    // sort things by length so that we try the shortest functions first.
    contents.sort_by_key(|x| x.1.len());

    for (name, content, failed) in &mut contents {
        if PRINT_EVERY_EXECUTION {
            println!("Running command over `{name}.ll`");
        }

        *failed = !run_command_for_each_fn(name, content);
    }

    for (name, _, _) in contents.into_iter().filter(|x| !x.2).take(30) {
        println!("Err: {name}");
    }
}

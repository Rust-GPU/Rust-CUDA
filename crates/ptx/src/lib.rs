pub mod lexer;
#[cfg(test)]
mod lexer_tests;
mod types;
// pub mod parser;

pub use types::*;

pub use ascii;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let string = std::fs::read_to_string("./simple_image.ptx").unwrap();
        let ascii_str = ascii::AsciiStr::from_ascii(&string).unwrap();
        let tokens = crate::lexer::Lexer::new(ascii_str)
            .map(|x| x.map(|x| x.kind))
            .collect::<Result<Vec<_>, _>>()
            .expect("expected no lexing error");
        println!("{:#?}", &tokens[tokens.len() - 40..]);
    }

    #[test]
    fn it_works2() {
        let ascii_str = ascii::AsciiStr::from_ascii("// Based on NVVM 7.0.1").unwrap();
        let tokens = crate::lexer::Lexer::new(ascii_str)
            .map(|x| x.map(|x| x.kind))
            .collect::<Result<Vec<_>, _>>()
            .expect("expected no lexing error");
        println!("{:#?}", tokens);
    }
}

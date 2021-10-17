use ptx::{ascii, lexer::Lexer};

static STRING: &'static str = include_str!("../simple_image.ptx");

fn main() {
    let tokens = Lexer::new(ascii::AsciiStr::from_ascii(STRING).unwrap())
        .map(|x| x.map(|x| x.kind))
        .collect::<Result<Vec<_>, _>>()
        .expect("expected no lexing error");
    println!("{}", tokens.len());
}

use ascii::AsciiStr;

use crate::{
    lexer::{Lexer, Token},
    types::*,
};
use std::iter::Peekable;

#[derive(Debug)]
pub(crate) struct TokenManager<'src> {
    pub(crate) token_pos: usize,
    pub(crate) lexer: Peekable<Lexer<'src>>,
}

pub type ParserResult<T> = Result<T, String>;

impl<'src> TokenManager<'src> {
    pub(crate) fn next(&mut self) -> Option<ParserResult<Token>> {
        let next = self.lexer.next()??;
        self.token_pos += 1;
        Some(Ok(next))
    }

    pub(crate) fn expect(&mut self, kind: TokenKind) -> ParserResult<Token> {
        match self.next() {
            None => Err(format!("Expected `{:?}`, but instead the file ended", kind)),
            Some(Ok(token)) if token.kind == kind => Ok(token),
            Some(Ok(token)) => Err(format!(
                "Expected `{:?}`, but instead found `{:?}`",
                kind, token.kind
            )),
            Some(Err(err)) => Err(err),
        }
    }

    pub(crate) fn cur_value(&mut self) -> TokenValue {
        self.lexer.values[self.token_pos]
    }
}

pub struct Parser<'src> {
    tokens: TokenManager<'src>,
}

impl<'src> Parser<'src> {
    pub fn new(src: &'src AsciiStr) -> Self {
        Self {
            tokens: TokenManager {
                lexer: Lexer::new(src),
                token_pos: 0,
            },
        }
    }

    fn directive(&mut self) -> ParserResult<>
}

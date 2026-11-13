//! A strict JSON parser for the inspector's tests: enough of JSON to check the events and the
//! exported snapshot, refusing NaN, infinity and raw control characters.

// Each test binary uses a different part of the parser.
#![allow(dead_code)]

/// A parsed JSON value: enough of JSON to check the events.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number.
    Num(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object, its members in order.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parses a whole document; an error says where it went wrong.
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser {
            bytes: text.as_bytes(),
            at: 0,
        };
        let value = p.value()?;
        p.space();
        if p.at != p.bytes.len() {
            return Err(format!("trailing text at byte {}", p.at));
        }
        Ok(value)
    }

    /// Member `key` of an object; panics if there is none.
    pub fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(members) => members
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("no member {key} in {self:?}")),
            _ => panic!("not an object: {self:?}"),
        }
    }

    /// Element `i` of an array.
    pub fn at(&self, i: usize) -> &Json {
        &self.arr()[i]
    }

    /// The elements of an array.
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            _ => panic!("not an array: {self:?}"),
        }
    }

    /// The value of a number.
    pub fn num(&self) -> f64 {
        match self {
            Json::Num(x) => *x,
            _ => panic!("not a number: {self:?}"),
        }
    }

    /// The value of a string.
    pub fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("not a string: {self:?}"),
        }
    }

    /// The value of a boolean.
    pub fn boolean(&self) -> bool {
        match self {
            Json::Bool(b) => *b,
            _ => panic!("not a boolean: {self:?}"),
        }
    }

    /// Whether this is `null`.
    pub fn is_null(&self) -> bool {
        *self == Json::Null
    }
}

/// A recursive-descent JSON parser over bytes.
struct Parser<'a> {
    /// The text.
    bytes: &'a [u8],
    /// The next byte to read.
    at: usize,
}

impl Parser<'_> {
    /// Skips white space.
    fn space(&mut self) {
        while self.at < self.bytes.len() && b" \t\r\n".contains(&self.bytes[self.at]) {
            self.at += 1;
        }
    }

    /// Reads `word` or fails.
    fn expect(&mut self, word: &str) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("expected {word} at byte {}", self.at))
        }
    }

    /// Reads one value.
    fn value(&mut self) -> Result<Json, String> {
        self.space();
        match self.bytes.get(self.at) {
            None => Err("unexpected end".to_string()),
            Some(b'n') => self.expect("null").map(|()| Json::Null),
            Some(b't') => self.expect("true").map(|()| Json::Bool(true)),
            Some(b'f') => self.expect("false").map(|()| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.bytes.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("bad array at byte {}", self.at)),
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.space();
                if self.bytes.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(Json::Obj(members));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    self.expect(":")?;
                    let value = self.value()?;
                    members.push((key, value));
                    self.space();
                    match self.bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Json::Obj(members));
                        }
                        _ => return Err(format!("bad object at byte {}", self.at)),
                    }
                }
            }
            Some(_) => self.number(),
        }
    }

    /// Reads a string, decoding its escapes.
    fn string(&mut self) -> Result<String, String> {
        self.expect("\"")?;
        let mut out = String::new();
        loop {
            let Some(&b) = self.bytes.get(self.at) else {
                return Err("unterminated string".to_string());
            };
            self.at += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.bytes[self.at];
                    self.at += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.bytes[self.at..self.at + 4])
                                .map_err(|e| e.to_string())?;
                            let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            self.at += 4;
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(format!("bad escape at byte {}", self.at)),
                    }
                }
                b if b < 0x20 => return Err(format!("raw control byte at {}", self.at)),
                _ => {
                    // Copy the whole UTF-8 sequence.
                    let start = self.at - 1;
                    let len = match b {
                        0xF0.. => 4,
                        0xE0.. => 3,
                        0xC0.. => 2,
                        _ => 1,
                    };
                    self.at = start + len;
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..self.at])
                            .map_err(|e| e.to_string())?,
                    );
                }
            }
        }
    }

    /// Reads a number in JSON's grammar (no NaN, no infinity, no leading `+`).
    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        if self.bytes.get(self.at) == Some(&b'-') {
            self.at += 1;
        }
        let digits = |p: &mut Self| {
            let s = p.at;
            while p.bytes.get(p.at).is_some_and(u8::is_ascii_digit) {
                p.at += 1;
            }
            p.at > s
        };
        if !digits(self) {
            return Err(format!("bad number at byte {start}"));
        }
        if self.bytes.get(self.at) == Some(&b'.') {
            self.at += 1;
            if !digits(self) {
                return Err(format!("bad fraction at byte {start}"));
            }
        }
        if matches!(self.bytes.get(self.at), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.bytes.get(self.at), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !digits(self) {
                return Err(format!("bad exponent at byte {start}"));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).unwrap();
        text.parse()
            .map(Json::Num)
            .map_err(|e| format!("{e} at byte {start}"))
    }
}

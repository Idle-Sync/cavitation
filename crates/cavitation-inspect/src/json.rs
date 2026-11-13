//! `JsonWriter`: a small JSON writer into a reused buffer, safe to embed in an HTML page.

use std::fmt::Write as _;

/// Writes JSON text into a buffer that is kept between uses, so that writing the same shape
/// of document again allocates nothing once the buffer has grown.
///
/// Commas are placed automatically: call [`key`](Self::key) before each value inside an
/// object, and write values one after another inside an array. The writer does not check
/// that the calls form a valid document; it is written by the library, for the library's
/// own events.
///
/// Two rules make its output safe wherever it lands:
///
/// - Numbers that are NaN or infinite are written as `null`, since JSON has no such values.
/// - Strings escape `"`, `\` and every control character, and also `<`, `>`, `&`, U+2028
///   and U+2029 (as `\u003c`, `\u003e`, `\u0026`, `\u2028`, `\u2029`), so the text can sit
///   inside an HTML `<script>` element without ending it, and inside JavaScript source.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::json::JsonWriter;
/// let mut w = JsonWriter::new();
/// w.begin_object()
///     .key("title")
///     .string("</script>")
///     .key("loss")
///     .number(f64::NAN)
///     .key("row")
///     .f32s(&[0.5, 1.0])
///     .end_object();
/// assert_eq!(w.as_str(), r#"{"title":"\u003c/script\u003e","loss":null,"row":[0.5,1.0]}"#);
/// ```
#[derive(Debug, Default, Clone)]
pub struct JsonWriter {
    /// The text written so far.
    buf: String,
    /// One entry per open object or array: whether a value has been written in it yet.
    open: Vec<bool>,
    /// Whether a key was just written, so the next value follows its colon with no comma.
    after_key: bool,
}

impl JsonWriter {
    /// An empty writer.
    ///
    /// ```
    /// let w = cavitation_inspect::json::JsonWriter::new();
    /// assert_eq!(w.as_str(), "");
    /// ```
    pub fn new() -> JsonWriter {
        JsonWriter::default()
    }

    /// Empties the writer, keeping its buffer's capacity.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.number(1.0);
    /// w.clear();
    /// assert_eq!(w.as_str(), "");
    /// ```
    pub fn clear(&mut self) {
        self.buf.clear();
        self.open.clear();
        self.after_key = false;
    }

    /// The text written so far.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.bool(false);
    /// assert_eq!(w.as_str(), "false");
    /// ```
    pub fn as_str(&self) -> &str {
        &self.buf
    }

    /// Writes the comma that separates this value from the previous one, if one is due.
    fn separate(&mut self) {
        if self.after_key {
            self.after_key = false;
            return;
        }
        if let Some(written) = self.open.last_mut() {
            if *written {
                self.buf.push(',');
            }
            *written = true;
        }
    }

    /// Opens an object.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.begin_object().end_object();
    /// assert_eq!(w.as_str(), "{}");
    /// ```
    pub fn begin_object(&mut self) -> &mut Self {
        self.separate();
        self.buf.push('{');
        self.open.push(false);
        self
    }

    /// Closes the innermost object.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.begin_object().key("a").uint(1).end_object();
    /// assert_eq!(w.as_str(), r#"{"a":1}"#);
    /// ```
    pub fn end_object(&mut self) -> &mut Self {
        self.open.pop();
        self.buf.push('}');
        self
    }

    /// Opens an array.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.begin_array().uint(1).uint(2).end_array();
    /// assert_eq!(w.as_str(), "[1,2]");
    /// ```
    pub fn begin_array(&mut self) -> &mut Self {
        self.separate();
        self.buf.push('[');
        self.open.push(false);
        self
    }

    /// Closes the innermost array.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.begin_array().end_array();
    /// assert_eq!(w.as_str(), "[]");
    /// ```
    pub fn end_array(&mut self) -> &mut Self {
        self.open.pop();
        self.buf.push(']');
        self
    }

    /// Writes an object key; the next call writes its value.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.begin_object().key("a<b").null().end_object();
    /// assert_eq!(w.as_str(), r#"{"a\u003cb":null}"#);
    /// ```
    pub fn key(&mut self, name: &str) -> &mut Self {
        self.separate();
        self.escaped(name);
        self.buf.push(':');
        self.after_key = true;
        self
    }

    /// Writes a string, escaped as the type's documentation describes.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.string("a \"b\"\n");
    /// assert_eq!(w.as_str(), r#""a \"b\"\n""#);
    /// ```
    pub fn string(&mut self, s: &str) -> &mut Self {
        self.separate();
        self.escaped(s);
        self
    }

    /// Appends `s` as a quoted, escaped JSON string.
    fn escaped(&mut self, s: &str) {
        self.buf.push('"');
        for c in s.chars() {
            match c {
                '"' => self.buf.push_str("\\\""),
                '\\' => self.buf.push_str("\\\\"),
                '\n' => self.buf.push_str("\\n"),
                '\r' => self.buf.push_str("\\r"),
                '\t' => self.buf.push_str("\\t"),
                '<' | '>' | '&' | '\u{2028}' | '\u{2029}' | '\u{0}'..='\u{1f}' => {
                    // Writing to a `String` cannot fail.
                    let _ = write!(self.buf, "\\u{:04x}", c as u32);
                }
                c => self.buf.push(c),
            }
        }
        self.buf.push('"');
    }

    /// Writes a number, or `null` when it is NaN or infinite. The text is the shortest that
    /// reads back as the same `f64`.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.begin_array().number(0.25).number(1e-7).number(f64::INFINITY).end_array();
    /// assert_eq!(w.as_str(), "[0.25,1e-7,null]");
    /// ```
    pub fn number(&mut self, x: f64) -> &mut Self {
        self.separate();
        if x.is_finite() {
            // `{:?}` gives the shortest round-trip form, in exponent notation when very
            // large or small, which is valid JSON either way.
            let _ = write!(self.buf, "{x:?}");
        } else {
            self.buf.push_str("null");
        }
        self
    }

    /// Writes an `f32` number, or `null` when it is NaN or infinite, in the shortest text
    /// that reads back as the same `f32`.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.f32(0.1);
    /// assert_eq!(w.as_str(), "0.1");
    /// ```
    pub fn f32(&mut self, x: f32) -> &mut Self {
        self.separate();
        if x.is_finite() {
            let _ = write!(self.buf, "{x:?}");
        } else {
            self.buf.push_str("null");
        }
        self
    }

    /// Writes an array of `f32` numbers, each as [`f32`](Self::f32) writes it.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.f32s(&[1.5, f32::NAN]);
    /// assert_eq!(w.as_str(), "[1.5,null]");
    /// ```
    pub fn f32s(&mut self, values: &[f32]) -> &mut Self {
        self.begin_array();
        for &x in values {
            self.f32(x);
        }
        self.end_array()
    }

    /// Writes an unsigned integer.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.uint(42);
    /// assert_eq!(w.as_str(), "42");
    /// ```
    pub fn uint(&mut self, x: u64) -> &mut Self {
        self.separate();
        let _ = write!(self.buf, "{x}");
        self
    }

    /// Writes a signed integer.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.int(-42);
    /// assert_eq!(w.as_str(), "-42");
    /// ```
    pub fn int(&mut self, x: i64) -> &mut Self {
        self.separate();
        let _ = write!(self.buf, "{x}");
        self
    }

    /// Writes `true` or `false`.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.bool(true);
    /// assert_eq!(w.as_str(), "true");
    /// ```
    pub fn bool(&mut self, x: bool) -> &mut Self {
        self.separate();
        self.buf.push_str(if x { "true" } else { "false" });
        self
    }

    /// Writes `null`.
    ///
    /// ```
    /// let mut w = cavitation_inspect::json::JsonWriter::new();
    /// w.null();
    /// assert_eq!(w.as_str(), "null");
    /// ```
    pub fn null(&mut self) -> &mut Self {
        self.separate();
        self.buf.push_str("null");
        self
    }
}

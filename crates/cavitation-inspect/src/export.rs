//! The inspector page, built into the crate, and the self-contained HTML export: the same
//! page with a snapshot of the run embedded and a policy that allows no connection.

/// The page served at `/`: one HTML file with its style and script inline.
pub(crate) const PAGE: &str = include_str!("../assets/inspector.html");

/// The Content Security Policy the page is served with: its own inline script and style,
/// images from `data:` URLs, connections only to this server, and no `<base>` element or
/// form submission.
pub(crate) const PAGE_POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'";

/// The policy of an exported document: the page's, with no connection allowed, since an
/// export is read from a file with no server behind it.
pub(crate) const EXPORT_POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'";

/// The element of the page after which the export's policy `<meta>` element goes: the first
/// in `<head>`, so it applies before any script. Matched without its line ending, so a page
/// checked out with CRLF line endings works the same.
const META_AFTER: &str = "<meta charset=\"utf-8\">";

/// The comment of the page that the export replaces with the embedded snapshot, just before
/// the page's script; matched without its line ending.
const SNAPSHOT_MARKER: &str = "<!-- snapshot -->";

/// Whether `needle` occurs in `haystack`. A `const fn`, so the check below runs when the
/// crate compiles.
const fn contains(haystack: &str, needle: &str) -> bool {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.len() > h.len() {
        return false;
    }
    let mut start = 0;
    while start + n.len() <= h.len() {
        let mut k = 0;
        while k < n.len() && h[start + k] == n[k] {
            k += 1;
        }
        if k == n.len() {
            return true;
        }
        start += 1;
    }
    false
}

// The page holds both places the export writes into, whatever its line endings: checked at
// compile time, so `document` cannot fail on the built-in page.
const _: () = assert!(
    contains(PAGE, META_AFTER) && contains(PAGE, SNAPSHOT_MARKER),
    "the inspector page must hold its charset element and its snapshot marker"
);

/// The page with `snapshot` (one JSON object, as [`Hub::export_json`] writes it) embedded
/// as `<script id="snapshot" type="application/json">`, which the page renders with its
/// controls off, and with [`EXPORT_POLICY`] in a `<meta>` element.
///
/// The snapshot comes from [`JsonWriter`](crate::json::JsonWriter), which escapes `<` in
/// every string, so no text in it can end the element. Any `<` left (none in valid output
/// of the writer, since JSON allows one only inside a string) is escaped here as well.
///
/// [`Hub::export_json`]: crate::sse::Hub::export_json
pub(crate) fn document(snapshot: &str) -> String {
    fill(PAGE, snapshot)
}

/// [`document`] for any `page` holding [`META_AFTER`] and [`SNAPSHOT_MARKER`]; adds no line
/// break of its own, so the page keeps whatever line endings it has.
fn fill(page: &str, snapshot: &str) -> String {
    let snapshot = snapshot.replace('<', "\\u003c");
    let meta = format!("<meta http-equiv=\"Content-Security-Policy\" content=\"{EXPORT_POLICY}\">");
    let element = format!("<script id=\"snapshot\" type=\"application/json\">{snapshot}</script>");
    let (head, rest) = page
        .split_once(META_AFTER)
        .expect("the charset element, checked when the crate compiles");
    let (middle, tail) = rest
        .split_once(SNAPSHOT_MARKER)
        .expect("the snapshot marker, checked when the crate compiles");
    let mut out = String::with_capacity(page.len() + meta.len() + element.len());
    out.push_str(head);
    out.push_str(META_AFTER);
    out.push_str(&meta);
    out.push_str(middle);
    out.push_str(&element);
    out.push_str(tail);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The page has each place the export writes into exactly once, before its script.
    #[test]
    fn page_has_each_marker_once() {
        assert_eq!(PAGE.matches(META_AFTER).count(), 1);
        assert_eq!(PAGE.matches(SNAPSHOT_MARKER).count(), 1);
        assert!(PAGE.find(META_AFTER).unwrap() < PAGE.find("<script").unwrap());
        assert!(PAGE.find(SNAPSHOT_MARKER).unwrap() < PAGE.find("<script").unwrap());
    }

    /// A stray `<` in the snapshot is escaped rather than written raw.
    #[test]
    fn stray_angle_bracket_is_escaped() {
        let html = document(r#"{"model":null,"x":"</script>"}"#);
        assert_eq!(html.matches("</script>").count(), 2);
        assert!(html.contains(r#""x":"\u003c/script>""#));
    }

    /// The page as a Windows checkout without the attributes rule would have it, with CRLF
    /// line endings, exports the same way: the policy before the script, the snapshot in
    /// its element, and no bare line feed added.
    #[test]
    fn crlf_page_exports() {
        let crlf = PAGE.replace('\n', "\r\n");
        let html = fill(&crlf, r#"{"model":null}"#);
        let meta =
            format!("<meta http-equiv=\"Content-Security-Policy\" content=\"{EXPORT_POLICY}\">");
        assert_eq!(html.matches(&meta).count(), 1);
        assert!(html.find(&meta).unwrap() < html.find("<script").unwrap());
        let open = r#"<script id="snapshot" type="application/json">"#;
        assert_eq!(html.matches(open).count(), 1);
        assert!(
            html.contains(
                r#"<script id="snapshot" type="application/json">{"model":null}</script>"#
            )
        );
        assert!(html.find(open).unwrap() < html.find("<script>").unwrap());
        assert!(!html.contains(SNAPSHOT_MARKER));
        assert_eq!(html.matches('\n').count(), html.matches("\r\n").count());
    }
}

//! The inspector page, built into the crate, and the self-contained HTML export: the same
//! page with a snapshot of the run embedded and a policy that allows no connection.

/// The page served at `/`: one HTML file with its style and script inline.
pub(crate) const PAGE: &str = include_str!("../assets/inspector.html");

/// The Content Security Policy the page is served with: its own inline script and style,
/// images from `data:` URLs, and connections only to this server.
pub(crate) const PAGE_POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'";

/// The policy of an exported document: the page's, with no connection allowed, since an
/// export is read from a file with no server behind it.
pub(crate) const EXPORT_POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'";

/// The line of the page after which the export's policy `<meta>` element goes: the first in
/// `<head>`, so it applies before any script.
const META_AFTER: &str = "<meta charset=\"utf-8\">\n";

/// The line of the page that the export replaces with the embedded snapshot, just before
/// the page's script.
const SNAPSHOT_MARKER: &str = "<!-- snapshot -->\n";

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
    let snapshot = snapshot.replace('<', "\\u003c");
    let meta =
        format!("<meta http-equiv=\"Content-Security-Policy\" content=\"{EXPORT_POLICY}\">\n");
    let element =
        format!("<script id=\"snapshot\" type=\"application/json\">{snapshot}</script>\n");
    let mut out = String::with_capacity(PAGE.len() + meta.len() + element.len());
    let (head, rest) = PAGE
        .split_once(META_AFTER)
        .expect("the page has its charset line");
    let (middle, tail) = rest
        .split_once(SNAPSHOT_MARKER)
        .expect("the page has its snapshot marker");
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

    /// The page has each place the export writes into exactly once.
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
}

# Package 26 retired semantic carriers

This directory preserves source history removed from production compilation by
ADR 0139 Package 26:

- `src/ir.rs`: textual pipeline IR superseded by ABIR nodes and plans.
- `src/source/bundle.rs`: owned uniform `SignalBundle` carrier superseded by
  validated `AbirDataset` roots plus borrowed views and `PayloadAccess`.

Files remain verbatim at retirement so audits and compatibility investigations
can inspect behavior without restoring public APIs. They are outside Cargo
module trees and must not be linked into production builds. Any temporary
converter belongs in an explicitly named legacy crate with a removal gate,
never in current source seams.

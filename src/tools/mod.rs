
pub mod constant_comment;
pub mod memory;
pub mod queued_comments;
pub mod switch;

pub use constant_comment::{ConstantCommentTracker, ConstantComment};
pub use memory::{MemoryAccessTracker, MemoryAccessState};
pub use queued_comments::{QueuedComments};

use crate::analysis::{AnalysisState};

/// Trait used when library state is needed to be included in Analysis.
/// Converts &mut reference from top-level user state to library state reference,
/// ideally just `{ &mut self.library_state_field }`
pub trait ProjectState<Dest>: AnalysisState {
    fn project(&mut self) -> &mut Dest;
}

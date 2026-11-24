//! Prediction processes: intra prediction (spec §8.5.1, [`intra`]) and inter prediction
//! (spec §8.5.2, [`inter`]). Both are called per plane by [`crate::tile::TileDecoder`]'s
//! residual path (`src/tile/residual.rs`).

mod inter;
mod intra;

#[cfg(test)]
pub(crate) use inter::block_inter_predict_scalar;
pub use inter::{predict_inter, InterPredictParams, RefPlaneView};
pub(crate) use inter::{MAX_BLOCK_DIM, MAX_INTERMEDIATE_HEIGHT};
pub use intra::predict_intra;
#[cfg(test)]
pub(crate) use intra::predict_intra_scalar;

#[cfg(test)]
use crate::framebuffer::Plane;
#[cfg(test)]
use crate::prob_tables::{DC_PRED, H_PRED, TM_PRED, V_PRED};

#[cfg(test)]
#[path = "../tests/unit/predict.rs"]
mod tests;

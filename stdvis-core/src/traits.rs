use std::io;

use anyhow::Result;
use ndarray::{ArrayViewD, ArrayViewMutD};

use crate::types::*;

/// A camera which captures images backed by a given `DataSource`.
pub trait Camera {
    type ImageStorage: ImageData;

    /// Returns the camera's config.
    fn config(&self) -> &CameraConfig;

    /// Grabs next image from the camera.
    fn grab_frame(&mut self) -> io::Result<Image<Self::ImageStorage>>;
}

/// A generalized format for image data.
/// `ImageData::Inner` represents the underlying data storage type.
pub trait ImageData {
    type Inner;

    /// Returns the image data as an array view of pixels.
    fn as_pixels(&self) -> ArrayViewD<u8>;

    /// Returns the image data as a mutable array view of pixels.
    fn as_pixels_mut(&mut self) -> ArrayViewMutD<u8>;

    /// Returns a reference to the underlying image data.
    fn as_raw(&self) -> &Self::Inner;

    /// Returns a mutable reference to the underlying image data.
    fn as_raw_mut(&mut self) -> &mut Self::Inner;
}

/// An interface that extracts contour groups from an `Image`.
pub trait ContourExtractor {
    fn extract_from<'src, I: ImageData>(
        &'src self,
        image: &Image<'src, I>,
    ) -> Result<Vec<ContourGroup<'src>>>;
}

/// An interface that computes a `VisionTarget` given a `ContourGroup`.
pub trait ContourAnalyzer {
    fn analyze(&self, contours: &ContourGroup) -> Result<VisionTarget>;
}

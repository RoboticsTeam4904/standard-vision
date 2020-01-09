use crate::types::*;
use ndarray::{ArrayView3, ArrayViewMut3};
use std::io;

/// A camera which captures images of type `I`.
pub trait Camera<I: ImageData> {
    /// Returns the camera's config.
    fn config(&self) -> &CameraConfig;

    /// Calibrates the camera.
    fn calibrate(&self) -> io::Result<()> {
        Ok(())
    }

    /// Grabs next image from the camera.
    fn grab_frame(&mut self) -> io::Result<Image<I>>;
}

/// A generalized format for image data.
/// `ImageData::Inner` represents the underlying data storage type.
pub trait ImageData {
    type Inner;

    /// Returns the image data as an array view of pixels.
    fn as_pixels(&self) -> ArrayView3<u8>;

    /// Returns the image data as a mutable array view of pixels.
    fn as_pixels_mut(&mut self) -> ArrayViewMut3<u8>;

    /// Returns a reference to the underlying image data.
    fn as_raw(&self) -> &Self::Inner;

    /// Returns a mutable reference to the underlying image data.
    fn as_raw_mut(&mut self) -> &mut Self::Inner;
}

/// An interface that extracts contours from an `Image`.
pub trait ContourExtractor<I: ImageData> {
    fn extract_from(&self, image: &Image<I>) -> Vec<Contour>;
}

/// An interface that computes `Target`s given a number of `Contour`s.
pub trait ContourAnalyzer {
    fn analyze(&self, config: &CameraConfig, contours: &Vec<Contour>) -> Vec<Target>;
}

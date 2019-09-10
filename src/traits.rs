use crate::types::*;
use std::io;
use std::error::Error;

pub trait Camera {
    fn config(&self) -> CameraConfig;
    fn calibrate(&self) -> io::Result<()>;
    fn grab_frame(&self) -> io::Result<Image>;
}

pub trait ContourExtractor {
    fn extract_from(&self, image: &Image) -> Vec<Contour>;
}

pub trait ContourAnalyzer {
    fn analyze(&self, contours: Vec<Contour>) -> Vec<Target>;
}

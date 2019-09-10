use crate::types::*;

pub trait Camera {
    fn calibrate(&self);
    fn config(&self) -> CameraConfig;
    fn grab_frame(&self) -> Image;
}

pub trait ContourExtractor {
    fn extract_from(&self, image: &Image) -> Vec<Contour>;
}

pub trait ContourAnalyzer {
    fn analyze(&self, contours: Vec<Contour>) -> Vec<Target>;
}

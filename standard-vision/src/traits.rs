use crate::types::*;
use std::io;

pub trait Camera {
    fn get_config(&self) -> &CameraConfig;
    fn calibrate(&self) -> io::Result<()> {
        Ok(())
    };
    fn grab_frame(&mut self) -> io::Result<Image>;
}

pub trait ContourExtractor {
    fn extract_from(&self, image: &Image) -> Vec<Contour>;
}

pub trait ContourAnalyzer {
    fn analyze(&self, contours: Vec<&Contour>) -> Vec<Target>;
}

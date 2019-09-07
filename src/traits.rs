use crate::types::*;

pub trait ContourExtractor {
    fn extract_from(&self, image: &Image) -> Vec<Contour>;
}

pub trait ContourAnalyzer {
    fn analyze(&self, contours: Vec<Contour>) -> Vec<Target>;
}

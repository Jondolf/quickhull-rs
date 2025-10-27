use crate::dim3::tri_face::TriFace;

pub fn validate_face_connectivity(face_index: u32, faces: &[TriFace]) {
    let face = &faces[face_index as usize];

    for i in 0..3 {
        let neighbor_index = face.neighbors[i];
        let neighbor = &faces[neighbor_index as usize];

        assert!(neighbor.valid, "Neighbor face is not valid.");
        assert_eq!(
            neighbor.neighbors[face.indirect_neighbors[i] as usize],
            face_index,
            "Neighbor face does not point back correctly. face_index: {}, neighbor_index: {}, i: {}",
            face_index, neighbor_index, i
        );
        assert_eq!(
            neighbor.indirect_neighbors[face.indirect_neighbors[i] as usize],
            i as u32,
            "Neighbor face indirect neighbor does not point back correctly. face_index: {}, neighbor_index: {}, i: {}",
            face_index, neighbor_index, i
        );
        assert_eq!(
            neighbor.first_point_from_edge(face.indirect_neighbors[i]),
            face.second_point_from_edge(i as u32),
            "Neighbor face edge points do not match. face_index: {}, neighbor_index: {}, i: {}",
            face_index,
            neighbor_index,
            i
        );
        assert_eq!(
            neighbor.second_point_from_edge(face.indirect_neighbors[i]),
            face.first_point_from_edge(i as u32),
            "Neighbor face edge points do not match. face_index: {}, neighbor_index: {}, i: {}",
            face_index,
            neighbor_index,
            i
        );
    }
}

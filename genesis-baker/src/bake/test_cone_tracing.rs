#[cfg(test)]
mod tests {
    use crate::bake::cone_tracing::calculate_v_attract;
    use crate::bake::spatial_grid::SpatialGrid;
    use crate::bake::neuron_placement::PlacedNeuron;
    use genesis_core::coords::pack_position;
    use glam::Vec3;

    fn make_neuron(x: u32, y: u32, z: u32, t: u8) -> PlacedNeuron {
        PlacedNeuron {
            position: pack_position(x, y, z, t as u32),
            type_idx: t as usize,
            layer_name: "Test".to_string(),
        }
    }

    #[test]
    fn test_single_target_directly_ahead() {
        let neurons = vec![make_neuron(0, 0, 5, 0)];
        let grid = SpatialGrid::new(&neurons, 10.0);
        
        let head_pos = Vec3::ZERO;
        let forward_dir = Vec3::Z;
        let fov_cos = 0.0; // 180 degrees (cos(90) = 0)
        let max_search_radius_vox = 10.0;
        
        let v_attract = calculate_v_attract(
            head_pos, forward_dir, fov_cos, max_search_radius_vox,
            &grid, &neurons, 0, usize::MAX, 1.0
        );

        // Expected to point directly at the target
        assert!((v_attract - Vec3::Z).length() < 1e-4);
    }

    #[test]
    fn test_target_outside_cone() {
        // Target is at +X
        let neurons = vec![make_neuron(10, 0, 0, 0)];
        let grid = SpatialGrid::new(&neurons, 20.0);
        
        let head_pos = Vec3::ZERO;
        let forward_dir = Vec3::Z; // Facing +Z
        let fov_cos = 0.866; // approx cos(30 degrees) = narrow cone
        let max_search_radius_vox = 20.0;
        
        let v_attract = calculate_v_attract(
            head_pos, forward_dir, fov_cos, max_search_radius_vox,
            &grid, &neurons, 0, usize::MAX, 1.0
        );

        // Target ignored due to FOV -> returns forward_dir
        assert_eq!(v_attract, forward_dir);
    }

    #[test]
    fn test_target_behind() {
        // Target is behind (-Z)
        let neurons = vec![make_neuron(0, 0, 0, 0)]; // Target at 0,0,0
        let grid = SpatialGrid::new(&neurons, 10.0);
        
        let head_pos = Vec3::new(0.0, 0.0, 5.0); // We are at Z=5
        let forward_dir = Vec3::Z; // Facing +Z (away from target)
        let fov_cos = 0.0; // 180 degree FOV (hemisphere forward)
        
        let v_attract = calculate_v_attract(
            head_pos, forward_dir, fov_cos, 10.0,
            &grid, &neurons, 0, usize::MAX, 1.0
        );

        // Target ignored due to FOV -> returns forward
        assert_eq!(v_attract, forward_dir);
    }

    #[test]
    fn test_two_targets_symmetric() {
        // Two targets equally to the left and right
        let _neurons = vec![
            make_neuron(2, 0, 5, 0), // Right
            make_neuron(0, 2, 5, 0), // Up (using Y as second symmetric axis)
        ];
        // Note: For pure 180 symmetry we'd use (-2,0,5) but u32 unsigned coords!
        // So we just use X and Y symmetrically while growing in Z.
        // Wait, 0, 0, 0 is origin, we can start at 5,5,0 and target 7,5,5 and 3,5,5.
        let neurons2 = vec![
            make_neuron(7, 5, 5, 0),
            make_neuron(3, 5, 5, 0),
        ];
        let grid = SpatialGrid::new(&neurons2, 20.0);
        
        let head_pos = Vec3::new(5.0, 5.0, 0.0);
        let forward_dir = Vec3::Z;
        
        let v_attract = calculate_v_attract(
            head_pos, forward_dir, 0.0, 20.0,
            &grid, &neurons2, 0, usize::MAX, 1.0
        );

        // Forces in X should cancel out exactly. V_attract should be pure Z.
        assert!(v_attract.x.abs() < 1e-5);
        assert!(v_attract.y.abs() < 1e-5);
        assert!(v_attract.z > 0.99); 
    }

    #[test]
    fn test_closer_target_wins() {
        let neurons = vec![
            make_neuron(7, 5, 5, 0),  // A: dx=2, dy=0, dz=5. Dist^2 = 29
            make_neuron(15, 5, 5, 0), // B: dx=10, dy=0, dz=5. Dist^2 = 125
        ];
        let grid = SpatialGrid::new(&neurons, 20.0);
        let head_pos = Vec3::new(5.0, 5.0, 0.0);
        let forward_dir = Vec3::Z;
        
        let v_attract = calculate_v_attract(
            head_pos, forward_dir, 0.0, 20.0,
            &grid, &neurons, 0, usize::MAX, 1.0
        );

        // It should lean right (positive X) because A (dx=2) is much closer than B (dx=10)
        // AND A pulls much stronger (1/29 vs 1/125).
        assert!(v_attract.x > 0.0);
    }

    #[test]
    fn test_type_affinity_zero() {
        let neurons = vec![make_neuron(0, 0, 5, 0)]; // Type 0
        let grid = SpatialGrid::new(&neurons, 10.0);
        
        // Owner is Type 0, affinity = 0.0
        let v_attract = calculate_v_attract(
            Vec3::ZERO, Vec3::Z, 0.0, 10.0,
            &grid, &neurons, 0, usize::MAX, 0.0 // Affinity 0.0 -> same type ignored
        );

        // Same type is ignored, returns forward
        assert_eq!(v_attract, Vec3::Z);
    }

    #[test]
    fn test_type_affinity_one() {
        let neurons = vec![
            make_neuron(5, 0, 5, 0), // Type 0 (Same)
            make_neuron(0, 5, 5, 1), // Type 1 (Diff)
        ];
        let grid = SpatialGrid::new(&neurons, 20.0);
        
        // Owner is Type 0, affinity = 1.0
        let v_attract = calculate_v_attract(
            Vec3::new(0.0, 0.0, 0.0), Vec3::Z, 0.0, 20.0,
            &grid, &neurons, 0, usize::MAX, 1.0 // Affinity 1.0 -> diff type ignored
        );

        // It should ONLY be attracted to Type 0 (X dir), and ignore Type 1 (Y dir)
        assert!(v_attract.x > 0.0);
        assert_eq!(v_attract.y, 0.0);
    }

    #[test]
    fn test_self_exclusion() {
        let neurons = vec![make_neuron(0, 0, 5, 0)]; // Target
        let grid = SpatialGrid::new(&neurons, 10.0);
        
        // Target is at idx 0, and our owner_soma_idx is ALSO 0
        let v_attract = calculate_v_attract(
            Vec3::ZERO, Vec3::Z, 0.0, 10.0,
            &grid, &neurons, 0, 0, 1.0 // owner = 0
        );

        // Target ignored because it's us
        assert_eq!(v_attract, Vec3::Z);
    }

    #[test]
    fn test_no_targets() {
        let grid = SpatialGrid::new(&[], 10.0);
        let v_attract = calculate_v_attract(
            Vec3::ZERO, Vec3::Z, 0.0, 10.0,
            &grid, &[], 0, usize::MAX, 1.0
        );
        assert_eq!(v_attract, Vec3::Z);
    }

    #[test]
    fn test_target_at_epsilon() {
        // Target exactly at head pos. Distance = 0.
        // Needs to not panic (e.g. div by zero or sqrt of negative)
        let neurons = vec![make_neuron(5, 5, 5, 0)];
        let grid = SpatialGrid::new(&neurons, 10.0);
        
        let head_pos = Vec3::new(5.0, 5.0, 5.0);
        
        // dist_sq < 1e-5 should trigger continue inside loop
        let v_attract = calculate_v_attract(
            head_pos, Vec3::Z, 0.0, 10.0,
            &grid, &neurons, 0, usize::MAX, 1.0
        );
        
        assert_eq!(v_attract, Vec3::Z);
    }
}

#[cfg(test)]
mod tests {
    use crate::bake::spatial_grid::SpatialGrid;
    use crate::bake::neuron_placement::PlacedNeuron;
    use genesis_core::coords::pack_position;
    use glam::Vec3;

    fn make_neuron(x: u32, y: u32, z: u32) -> PlacedNeuron {
        PlacedNeuron {
            position: pack_position(x, y, z, 0),
            type_idx: 0,
            layer_name: "Test".to_string(),
        }
    }

    #[test]
    fn test_single_neuron_found() {
        let neurons = vec![make_neuron(5, 5, 5)];
        let grid = SpatialGrid::new(&neurons, 2.0);
        let found = grid.get_in_radius(Vec3::new(5.0, 5.0, 5.0), 1.0);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], 0);
    }

    #[test]
    fn test_out_of_radius() {
        let neurons = vec![make_neuron(0, 0, 0)];
        let grid = SpatialGrid::new(&neurons, 2.0);
        let found = grid.get_in_radius(Vec3::new(100.0, 100.0, 100.0), 3.0);
        assert!(found.is_empty());
    }

    #[test]
    fn test_radius_boundary() {
        let neurons = vec![make_neuron(4, 0, 0)];
        let grid = SpatialGrid::new(&neurons, 2.0);
        
        // 4 units away. If cell size is 2.0, requesting from (0,0,0) with r=3.9
        // will check cx from flooor(-3.9/2)=-2 to floor(3.9/2)=1.
        // The neuron is at cx = 4/2 = 2. It will NOT be checked.
        let found1 = grid.get_in_radius(Vec3::ZERO, 3.9);
        assert!(found1.is_empty());

        // With r=5.0, max_cx = floor(5/2) = 2. It WILL be checked.
        let found2 = grid.get_in_radius(Vec3::ZERO, 5.0);
        assert_eq!(found2.len(), 1);
        assert_eq!(found2[0], 0);
    }

    #[test]
    fn test_multiple_cells() {
        let neurons = vec![
            make_neuron(0, 0, 0),
            make_neuron(5, 5, 5),
            make_neuron(10, 10, 10),
        ];
        let grid = SpatialGrid::new(&neurons, 2.0);
        let found = grid.get_in_radius(Vec3::new(5.0, 5.0, 5.0), 10.0);
        assert_eq!(found.len(), 3); // All should be found
    }

    #[test]
    fn test_empty_grid() {
        let grid = SpatialGrid::new(&[], 2.0);
        let found = grid.get_in_radius(Vec3::ZERO, 100.0);
        assert!(found.is_empty());
    }

    #[test]
    fn test_same_cell_multiple() {
        let neurons = vec![
            make_neuron(1, 1, 1),
            make_neuron(1, 1, 1),
            make_neuron(1, 1, 1),
        ];
        let grid = SpatialGrid::new(&neurons, 2.0);
        let found = grid.get_in_radius(Vec3::new(1.0, 1.0, 1.0), 1.0);
        assert_eq!(found.len(), 3);
        
        // They should be indices 0, 1, 2 (order not guaranteed by spec, but usually sequential)
        assert!(found.contains(&0));
        assert!(found.contains(&1));
        assert!(found.contains(&2));
    }
}

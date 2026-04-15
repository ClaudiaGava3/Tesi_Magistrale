#!/usr/bin/env python3
"""
Simple test script for 2D drone model and solver verification.
"""

import numpy as np
import sys

# Add src to path
sys.path.insert(0, '/home/claudia/TESI/VBOC-sth-dev-obs/src')

from vboc.parser import Parameters
from vboc.abstract import Model
from vboc.controller import ViabilityController

def test_model_dimensions():
    """Test that the model has the correct state/input dimensions."""
    print("=" * 60)
    print("TEST 1: Model Dimensions")
    print("=" * 60)
    
    try:
        params = Parameters('sth')
        model = Model(params)
        
        print(f"✓ Model created successfully")
        print(f"  - State dimension (nx): {model.nx} (expected 8)")
        print(f"  - Input dimension (nu): {model.nu} (expected 2)")
        print(f"  - Output dimension (ny): {model.ny} (expected 10)")
        print(f"  - Generalized coords (nq): {model.nq} (expected 2)")
        print(f"  - Velocity dimension (nv): {model.nv} (expected 2)")
        print(f"  - Box dims (nbox): {model.nbox} (expected 4)")
        print(f"  - Num constraints (ncon): {model.ncon} (expected 4)")
        
        # Verify dimensions
        assert model.nx == 8, f"Expected nx=8, got {model.nx}"
        assert model.nu == 2, f"Expected nu=2, got {model.nu}"
        assert model.nq == 2, f"Expected nq=2, got {model.nq}"
        assert model.nv == 2, f"Expected nv=2, got {model.nv}"
        assert model.nbox == 4, f"Expected nbox=4, got {model.nbox}"
        
        print("\n✓ All dimension checks passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_input_bounds():
    """Test that input bounds are correctly set."""
    print("=" * 60)
    print("TEST 2: Input Bounds")
    print("=" * 60)
    
    try:
        params = Parameters('sth')
        model = Model(params)
        
        print(f"✓ Input bounds loaded")
        print(f"  - u_min: {model.u_min}")
        print(f"  - u_max: {model.u_max}")
        
        # Verify bounds
        assert len(model.u_min) == 2, f"Expected u_min length 2, got {len(model.u_min)}"
        assert len(model.u_max) == 2, f"Expected u_max length 2, got {len(model.u_max)}"
        assert np.all(model.u_min >= 0), "u_min should be >= 0 (thrust forces)"
        assert np.all(model.u_max > model.u_min), "u_max should be > u_min"
        
        print("\n✓ All input bound checks passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_box_constraints():
    """Test that box constraints are properly defined."""
    print("=" * 60)
    print("TEST 3: Box Constraints [0, 1000]")
    print("=" * 60)
    
    try:
        params = Parameters('sth')
        model = Model(params)
        
        print(f"✓ Box constraints loaded")
        print(f"  - Environment min: {model.env_min}")
        print(f"  - Environment max: {model.env_max}")
        print(f"  - Num constraints: {model.con_h_expr.size()[0]}")
        print(f"  - Constraint bounds: {model.env_dimensions}")
        
        # Verify constraints
        assert model.env_min == 0, f"Expected env_min=0, got {model.env_min}"
        assert model.env_max == 1000, f"Expected env_max=1000, got {model.env_max}"
        assert model.con_h_expr.size()[0] == 4, f"Expected 4 constraints, got {model.con_h_expr.size()[0]}"
        
        print("\n✓ All box constraint checks passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_controller_creation():
    """Test that controller can be created from model."""
    print("=" * 60)
    print("TEST 4: Controller Creation")
    print("=" * 60)
    
    try:
        params = Parameters('sth')
        model = Model(params)
        controller = ViabilityController(model)
        
        print(f"✓ Controller created successfully")
        print(f"  - OCP name: {controller.ocp_name}")
        print(f"  - Horizon (N): {controller.N}")
        print(f"  - Time step (dt): {params.dt}")
        print(f"  - Total time: {controller.N * params.dt:.3f}s")
        
        print("\n✓ Controller creation passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  2D DRONE MODEL AND SOLVER VERIFICATION TESTS".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    results.append(("Model Dimensions", test_model_dimensions()))
    results.append(("Input Bounds", test_input_bounds()))
    results.append(("Box Constraints", test_box_constraints()))
    results.append(("Controller Creation", test_controller_creation()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

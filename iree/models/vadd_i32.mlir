// Simple vector add model (integer) for IREE bare-metal testing with custom accelerator
// Input: A[16xi32], B[16xi32] -> element-wise add -> C[16xi32]

module @vadd_i32 {
  func.func @forward(%A: tensor<16xi32>, %B: tensor<16xi32>) -> tensor<16xi32> {
    // Element-wise addition: C = A + B
    %C_init = tensor.empty() : tensor<16xi32>
    %C = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%A, %B : tensor<16xi32>, tensor<16xi32>) outs(%C_init : tensor<16xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %add = arith.addi %a, %b : i32
      linalg.yield %add : i32
    } -> tensor<16xi32>

    return %C : tensor<16xi32>
  }
}

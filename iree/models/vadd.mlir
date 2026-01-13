// Simple vector add model for IREE bare-metal testing with custom accelerator
// Input: A[16xf32], B[16xf32] -> element-wise add -> C[16xf32]

module @vadd {
  func.func @forward(%A: tensor<16xf32>, %B: tensor<16xf32>) -> tensor<16xf32> {
    // Element-wise addition: C = A + B
    %C_init = tensor.empty() : tensor<16xf32>
    %C = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%A, %B : tensor<16xf32>, tensor<16xf32>) outs(%C_init : tensor<16xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %add = arith.addf %a, %b : f32
      linalg.yield %add : f32
    } -> tensor<16xf32>

    return %C : tensor<16xf32>
  }
}

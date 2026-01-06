// Simple matmul + softmax model for IREE bare-metal testing
// Input: A[4x8], B[8x4] -> matmul -> C[4x4] -> softmax -> output[4x4]

module @matmul_softmax {
  func.func @forward(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>) -> tensor<4x4xf32> {
    // Initialize output tensor with zeros
    %cst = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4x4xf32>
    %C_init = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>

    // Matmul: C = A @ B
    %C = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%C_init : tensor<4x4xf32>) -> tensor<4x4xf32>

    // Softmax along last dimension (axis=1)
    // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    // Step 1: Find max along axis 1
    %neg_inf = arith.constant 0xFF800000 : f32  // -inf
    %max_init = tensor.empty() : tensor<4xf32>
    %max_fill = linalg.fill ins(%neg_inf : f32) outs(%max_init : tensor<4xf32>) -> tensor<4xf32>
    %max = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%C : tensor<4x4xf32>) outs(%max_fill : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.maximumf %in, %out : f32
      linalg.yield %m : f32
    } -> tensor<4xf32>

    // Step 2: Subtract max and compute exp
    %exp_init = tensor.empty() : tensor<4x4xf32>
    %exp = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%C, %max : tensor<4x4xf32>, tensor<4xf32>) outs(%exp_init : tensor<4x4xf32>) {
    ^bb0(%c: f32, %m: f32, %out: f32):
      %sub = arith.subf %c, %m : f32
      %e = math.exp %sub : f32
      linalg.yield %e : f32
    } -> tensor<4x4xf32>

    // Step 3: Sum of exp along axis 1
    %zero = arith.constant 0.000000e+00 : f32
    %sum_init = tensor.empty() : tensor<4xf32>
    %sum_fill = linalg.fill ins(%zero : f32) outs(%sum_init : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%exp : tensor<4x4xf32>) outs(%sum_fill : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %s = arith.addf %in, %out : f32
      linalg.yield %s : f32
    } -> tensor<4xf32>

    // Step 4: Divide by sum
    %result_init = tensor.empty() : tensor<4x4xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%exp, %sum : tensor<4x4xf32>, tensor<4xf32>) outs(%result_init : tensor<4x4xf32>) {
    ^bb0(%e: f32, %s: f32, %out: f32):
      %div = arith.divf %e, %s : f32
      linalg.yield %div : f32
    } -> tensor<4x4xf32>

    return %result : tensor<4x4xf32>
  }
}

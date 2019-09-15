#include <cstdio>
#include <cstdlib>

extern "C" {
void ApplyTransformation(float* kx, float* ky, int nkeys, float* tform) {

#define tformAt(i,j)  (tform[i*3+j])
    
  for (int i = 0; i < nkeys; i++) {
    float denominator = kx[i] * tformAt(2, 0) + ky[i] * tformAt(2, 1) + tformAt(2, 2);
    float kx_ =         kx[i] * tformAt(0, 0) + ky[i] * tformAt(0, 1) + tformAt(0, 2);
    float ky_ =         kx[i] * tformAt(1, 0) + ky[i] * tformAt(1, 1) + tformAt(1, 2);

//    printf("denominator = %f\n", denominator);
//    printf("old: kx = %.02f, ky = %.02f\n", kx[i], ky[i]);
    kx[i] = kx_ / denominator;
    ky[i] = ky_ / denominator;
//    printf("kx = %.02f, ky = %.02f\n", kx[i], ky[i]);
  }
}

// void DistanceBasedMatching2(float* kx1, float* ky1, int nkeys1,
//                            float* kx2, float* ky2, int nkeys2,
//                            float* tform,
//                            int* nn_idx, float* nn_dist) {

// //  for (int i = 0; i < 9; i++) {
// //    printf("%f\n", tform[i]);
// //    printf("kx = %.02f, ky = %.02f\n", kx1[i], ky1[i]);
// //  }
//   ApplyTransformation(kx1, ky1, nkeys1, tform);

//   float dx, dy, dist;


// //  float* t = (float*)malloc(nkeys1 * nkeys2 * sizeof(float));
//   float* t = new float[nkeys1 * nkeys2 * sizeof(float)];

//   for (int i = 0; i < nkeys1; i++) {
//     for (int j = 0; j < nkeys2; j++) {
//       t[i * nkeys2 + j] = 9999999.9f;
//     }
//   }

//   int* min_idx1 = new int[nkeys1];
//   int* min_idx2 = new int[nkeys2];

//   for (int i = 0; i < nkeys1; i++) {
//     nn_dist[i] = 0.0f;
//     nn_idx[i] = -1;
    
//     float min_d = 9999999.9f;
//     int min_idx = -1;
//     for (int j = 0; j < nkeys2; j++) {
//       dx = kx1[i] - kx2[j];
//       dy = ky1[i] - ky2[j];
//       dist = dx * dx + dy * dy;
//       if (dist < min_d) {
//         min_d = dist;
//         min_idx = j;
//       }
//     }
//     t[i * nkeys2 + min_idx] = min_d;
    
//     nn_dist[i] = min_d;
//     nn_idx[i] = min_idx;
//   }


// //   for (int j = 0; j < nkeys2; j++) {
// //     // find min
// //     float min_d = 9999999.9f;
// //     int min_idx = -1;
// //     for (int i = 0; i < nkeys1; i++) {
// //       if (t[i * nkeys2 + j] < min_d) {
// //         min_d = t[i * nkeys2 + j];
// //         min_idx = i;
// //       }
// //     }
// //
// //     if (min_idx >= 0) {
// //       nn_dist[min_idx] = min_d;
// //       nn_idx[min_idx] = j;
// //     }
// //   }

// //  free(t);
//   delete[] t;
// }


void DistanceBasedMatching(float* kx1, float* ky1, int nkeys1,
                           float* kx2, float* ky2, int nkeys2,
                           float* tform,
                           int* nn_idx, float* nn_dist) {

  ApplyTransformation(kx1, ky1, nkeys1, tform);

  // compute distance table
  float* t = new float[nkeys1 * nkeys2];
  float dx, dy, dist;
  for (int i = 0; i < nkeys1; i++) {
    for (int j = 0; j < nkeys2; j++) {
      dx = kx1[i] - kx2[j];
      dy = ky1[i] - ky2[j];
      dist = dx * dx + dy * dy;
      t[i * nkeys2 + j] = dist;
    }
  }

  int* min_idx1 = new int[nkeys1];
  int* min_idx2 = new int[nkeys2];

  // find nearest neighbor 1
  float md, midx;
  for (int i = 0; i < nkeys1; i++) {
    md = 99999999.9f;
    midx = -1;
    for (int j = 0; j < nkeys2; j++) {
      if (t[i * nkeys2 + j] < md) {
        md = t[i * nkeys2 + j];
        midx = j;
      }
    }
    min_idx1[i] = midx;
    nn_dist[i] = md;
  }

  // find nearest neighbor 2
  for (int j = 0; j < nkeys2; j++) {
    md = 99999999.9f;
    midx = -1;
    for (int i = 0; i < nkeys1; i++) {
      if (t[i * nkeys2 + j] < md) {
        md = t[i * nkeys2 + j];
        midx = i;
      }
    }
    min_idx2[j] = midx;
  }

  // bidirectional check
  for (int i = 0; i < nkeys1; i++) {
    if (i == min_idx2[min_idx1[i]]) {
      nn_idx[i] = min_idx1[i];
    } else {
      nn_idx[i] = -1;
      nn_dist[i] = 99999999.9f;
    }
  }

  // no bidirectional check
//  for (int i = 0; i < nkeys1; i++) {
//    nn_idx[i] = min_idx1[i];
//  }

  delete[] t;
  delete[] min_idx1;
  delete[] min_idx2;
}






}
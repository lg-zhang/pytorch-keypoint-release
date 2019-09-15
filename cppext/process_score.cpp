#include <cstdio>
#include <cmath>

extern "C" {

void NonExtremaSuppression(float* score, int w, int h, unsigned char* is_extrema, int choose_maxima) {
  float v;
  float* pt;
  
  for (int y = 1; y < h - 1; y++) {
    for (int x = 1; x < w - 1; x++) {
      pt = (score + x + y * w);
      v = *pt;
       
#define CHECK_NEIGHBORS_2D(CMP)      \
  ( v CMP *(pt + 1 + 0) &&        \
    v CMP *(pt - 1 + 0) &&        \
    v CMP *(pt + 0 + w) &&        \
    v CMP *(pt + 0 - w) &&        \
    v CMP *(pt + 1 - w) &&        \
    v CMP *(pt - 1 - w) &&        \
    v CMP *(pt + 1 + w) &&        \
    v CMP *(pt - 1 + w) )

      if (choose_maxima > 0 && CHECK_NEIGHBORS_2D(>)) {
        is_extrema[x + y * w] = 1;
      } else if (choose_maxima <= 0  && CHECK_NEIGHBORS_2D(<)) {
        is_extrema[x + y * w] = 1;
      }
    }
  }
}


void NonExtremaSuppressionMultiScale(float* score, int c, int w, int h, unsigned char* is_extrema, int choose_maxima) {
  float v;
  float* pt;
  int offset;

  int xo = 1; // x-stride
  int yo = w; // y-stride
  int so = w * h; // s-stride

  for (int s = 1; s < c - 1; s++) {
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        offset = x * xo + y * yo + s * so;
        pt = (score + offset);
        v = *pt;
        
  #define CHECK_NEIGHBORS(CMP)                      \
        ( v CMP *(pt + xo) &&                       \
          v CMP *(pt - xo) &&                       \
          v CMP *(pt + so) &&                       \
          v CMP *(pt - so) &&                       \
          v CMP *(pt + yo) &&                       \
          v CMP *(pt - yo) &&                       \
                                                    \
          v CMP *(pt + yo + xo) &&                  \
          v CMP *(pt + yo - xo) &&                  \
          v CMP *(pt - yo + xo) &&                  \
          v CMP *(pt - yo - xo) &&                  \
                                                    \
          v CMP *(pt + xo      + so) &&             \
          v CMP *(pt - xo      + so) &&             \
          v CMP *(pt + yo      + so) &&             \
          v CMP *(pt - yo      + so) &&             \
          v CMP *(pt + yo + xo + so) &&             \
          v CMP *(pt + yo - xo + so) &&             \
          v CMP *(pt - yo + xo + so) &&             \
          v CMP *(pt - yo - xo + so) &&             \
                                                    \
          v CMP *(pt + xo      - so) &&             \
          v CMP *(pt - xo      - so) &&             \
          v CMP *(pt + yo      - so) &&             \
          v CMP *(pt - yo      - so) &&             \
          v CMP *(pt + yo + xo - so) &&             \
          v CMP *(pt + yo - xo - so) &&             \
          v CMP *(pt - yo + xo - so) &&             \
          v CMP *(pt - yo - xo - so) )


        if ((choose_maxima > 0 && CHECK_NEIGHBORS(>)) ||
            (choose_maxima <= 0  && CHECK_NEIGHBORS(<))) {
          *(is_extrema + offset) = 1;
        }
      }
    }
  }
}



// modified from vlfeat
void CalcSubpixel(float* score, int w, int h, int n_keys, float* keys_x, float* keys_y) {
//  printf("subpix\n");
  int x, y, dx, dy;
  double Dx, Dy, Dxx, Dyy, Dxy;
  double A[4];
  double b[2];
  int i, j, ii, jj;

  for (int k = 0; k < n_keys; k++) {
    // start from the current pixel (may shift, depending on dx, dy)
    x = (int)keys_x[k]; 
    y = (int)keys_y[k];
    
    dx = 0;
    dy = 0;

    for (int iter = 0 ; iter < 5 ; ++iter) {
      x += dx ;
      y += dy ;

      float* pt = score + x + w * y;
      /** @brief Index GSS @internal */
#define at(dx,dy) (*(pt + dx + dy * w))

      /** @brief Index matrix A @internal */
#define Aat(i,j)  (A[(i)+(j)*2])

      /* compute the gradient */
      Dx = 0.5 * (at(+1,0) - at(-1,0)) ;
      Dy = 0.5 * (at(0,+1) - at(0,-1)) ;

      /* compute the Hessian */
      Dxx = (at(+1,0) + at(-1,0) - 2.0 * at(0,0)) ;
      Dyy = (at(0,+1) + at(0,-1) - 2.0 * at(0,0)) ;
      
      Dxy = 0.25 * ( at(+1,+1) + at(-1,-1) - at(-1,+1) - at(+1,-1) ) ;

      /* solve linear system ....................................... */
      Aat(0,0) = Dxx ;
      Aat(1,1) = Dyy ;
      Aat(0,1) = Aat(1,0) = Dxy ;

      b[0] = - Dx ;
      b[1] = - Dy ;

      /* Gauss elimination */
      for(j = 0 ; j < 2 ; ++j) {
        double maxa    = 0 ;
        double maxabsa = 0 ;
        int    maxi    = -1 ;
        double tmp ;

        /* look for the maximally stable pivot */
        for (i = j ; i < 2 ; ++i) {
          double a    = Aat (i,j) ;
          double absa = fabs (a) ;
          if (absa > maxabsa) {
            maxa    = a ;
            maxabsa = absa ;
            maxi    = i ;
          }
        }

        /* if singular give up */
        if (maxabsa < 1e-10f) {
          b[0] = 0 ;
          b[1] = 0 ;
//          printf("singular\n");
          break ;
        }

        i = maxi ;

        /* swap j-th row with i-th row and normalize j-th row */
        for(jj = j ; jj < 2 ; ++jj) {
          tmp = Aat(i,jj) ; Aat(i,jj) = Aat(j,jj) ; Aat(j,jj) = tmp ;
          Aat(j,jj) /= maxa ;
        }
        tmp = b[j] ; b[j] = b[i] ; b[i] = tmp ;
        b[j] /= maxa ;

        /* elimination */
        for (ii = j+1 ; ii < 2 ; ++ii) {
          double x = Aat(ii,j) ;
          for (jj = j ; jj < 2 ; ++jj) {
            Aat(ii,jj) -= x * Aat(j,jj) ;
          }
          b[ii] -= x * b[j] ;
        }
      }

      /* backward substitution */
      for (i = 1 ; i > 0 ; --i) {
        double x = b[i] ;
        for (ii = i-1 ; ii >= 0 ; --ii) {
          b[ii] -= x * Aat(ii,i) ;
        }
      }

      /* .......................................................... */
      /* If the translation of the keypoint is big, move the keypoint
       * and re-iterate the computation. Otherwise we are all set.
       */

      dx= ((b[0] >  0.6 && x < w - 2) ?  1 : 0)
        + ((b[0] < -0.6 && x > 1    ) ? -1 : 0) ;

      dy= ((b[1] >  0.6 && y < h - 2) ?  1 : 0)
        + ((b[1] < -0.6 && y > 1    ) ? -1 : 0) ;

      if (dx == 0 && dy == 0) break ;
    }  

//    printf("b = [%.04f, %.04f]\n", b[0], b[1]);
    keys_x[k] = (float)x + b[0] ;
    keys_y[k] = (float)y + b[1] ;
  }
}

}
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;
typedef vector<vector<double>> matrix;

void build_up_b(int nx, int ny,
    matrix &b, double rho, double dt,
    matrix &u, matrix &v, double dx, double dy) {

  for(int i=1; i<ny-1; i++) {
    for(int j=1; j<nx-1; j++) {
      b[i][j] = rho * (1 / dt *
            ((u[i][j+1] - u[i][j-1]) /
             (2. * dx) + (v[i+1][j] - v[i-1][j]) / (2. * dy)) -
            pow((u[i][j+1] - u[i][j-1]) / (2. * dx), 2) -
              2. * ((u[i+1][j] - u[i-1][j]) / (2. * dy) *
                   (v[i][j+1] - v[i][j-1]) / (2. * dx)) -
                   pow((v[i+1][j] - v[i-1][j]) / (2. * dy), 2));
    }
  }
}

int NIT = 50;
void pressure_poisson(int nx, int ny, matrix &p,
    double dx, double dy, matrix &b) {

  matrix pn(ny,vector<double>(nx));
  for(int q=0; q<NIT; q++) {
    for(int i=0; i<ny; i++) {
      for(int j=0; j<nx; j++){
        pn[i][j] = p[i][j];
      }
    }
    for(int i=1; i<ny-1; i++) {
      for(int j=1; j<nx-1; j++){
        p[i][j] =  ((pn[i][j+1] + pn[i][j-1]) * dy*dy +
                    (pn[i+1][j] + pn[i-1][j]) * dx*dx) /
                    (2 * (dx*dx + dy*dy)) -
                    dx*dx * dy*dy / (2 *(dx*dx + dy*dy)) *
                    b[i][j];
      }
    }
    for(int i=0; i<ny; i++) {
      p[i][nx-1] = p[i][nx-2];
      p[i][0] = p[i][1];
    }
    for(int i=0; i<nx; i++) {
      p[0][i] = p[1][i];
      p[ny-1][i] = 0;
    }
  }
}

int cavity_flow(double eps, int nx, int ny,
    matrix &u, matrix &v, double dt, double dx,
    double dy, matrix &p, double rho, double nu) {

  matrix un(ny,vector<double>(nx));
  matrix vn(ny,vector<double>(nx));
  matrix b(ny,vector<double>(nx));

  int nt = 0;
  double udiff = 233;
  for(;udiff > eps; nt++) {
    for(int i=0; i<ny; i++) {
      for(int j=0; j<nx; j++){
        un[i][j] = u[i][j];
        vn[i][j] = v[i][j];
      }
    }
    build_up_b(nx, ny, b, rho, dt, u, v, dx, dy);
    pressure_poisson(nx, ny, p, dx, dy, b);
    for(int i=1; i<ny-1; i++) {
      for(int j=1; j<nx-1; j++) {
        u[i][j] =  un[i][j]-
                   un[i][j] * dt / dx *
                  (un[i][j] - un[i][j-1]) -
                   vn[i][j] * dt / dy *
                  (un[i][j] - un[i-1][j]) -
                   dt / (2. * rho * dx) * (p[i][j+1] - p[i][j-1]) +
                   nu * (dt / (dx*dx) *
                  (un[i][j+1] - 2. * un[i][j] + un[i][j-1]) +
                   dt / (dy*dy) *
                  (un[i+1][j] - 2. * un[i][j] + un[i-1][j]));
      }
    }
    for(int i=1; i<ny-1; i++) {
      for(int j=1; j<nx-1; j++) {
        v[i][j] = vn[i][j] -
                  un[i][j] * dt / dx *
                 (vn[i][j] - vn[i][j-1]) -
                  vn[i][j] * dt / dy *
                 (vn[i][j] - vn[i-1][j]) -
                  dt / (2. * rho * dy) * (p[i+1][j] - p[i-1][j]) +
                  nu * (dt / (dx*dx) *
                 (vn[i][j+1] - 2. * vn[i][j] + vn[i][j-1]) +
                  dt / (dy*dy) *
                 (vn[i+1][j] - 2. * vn[i][j] + vn[i-1][j]));
      }
    }

    for(int i=0; i<ny; i++) {
      u[i][0] = 0;
      u[i][nx-1] = 0;
      v[i][0] = 0;
      v[i][nx-1] = 0;
    }
    for(int i=0; i<nx; i++) {
      u[0][i] = 0;
      u[ny-1][i] = 1;
      v[0][i] = 0;
      v[ny-1][i] = 0;
    }

    udiff = 0;
    double usum = 0;
    for (int i=0; i<ny; i++){
      for(int j=0; j<nx; j++){
        udiff += abs(u[i][j] - un[i][j]);
        usum += abs(u[i][j]);
      }
    }
    udiff /= usum;
  }
  return nt;
}


int main() {
  int nx = 41, ny = 41;
  double dx = 2. / (nx - 1.);
  double dy = 2. / (ny - 1.);
  double dt = .001;
  double rho = 1., nu = .1;

  matrix u(ny,vector<double>(nx));
  matrix v(ny,vector<double>(nx));
  matrix p(ny,vector<double>(nx));
  for(int i=0; i<ny; i++) {
    for(int j=0; j<nx; j++){
      u[i][j] = 0;
      v[i][j] = 0;
      p[i][j] = 0;
    }
  }

  double eps = .0000001;
  int nt;
  auto tic = chrono::steady_clock::now();
  nt = cavity_flow(eps, nx, ny, u, v, dt, dx, dy, p, rho, nu);
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("Reaching convergence after %d steps.\n", nt);
  printf("Elapsed time is %lf s.\n",time);

  double ures = 0, vres = 0, pres = 0;
  for(int i=0; i<ny; i++) {
    for(int j=0; j<nx; j++) {
      ures += abs(u[i][j]);
      vres += abs(v[i][j]);
      pres += abs(p[i][j]);
    }
  }
  printf("Sum(|u|)=%f\n", ures);
  printf("Sum(|v|)=%f\n", vres);
  printf("Sum(|p|)=%f\n", pres);

  return 0;
}

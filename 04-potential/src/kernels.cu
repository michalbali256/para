#include "kernels.h"



/*
 * Sample Kernel
 */
__global__ void my_kernel(Point<double> * in_points, Point<double>* out_points)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	point_t force(0.0, 0.0);

	for (index_t j = 0; j < points.size(); ++j)
		addRepulsiveForce(in_points, i, j, force);

	for (size_t j = 0; j < neighbors_[i].count; ++j)
		addCompulsiveForce(in_points, i, neighbors_[i].neigh[j].neigh, neighbors_[i].neigh[j].length, force);

	double vel_x = (mVelocities[i].x + (real_t)force.x * velocity_update_fact) * this->mParams.slowdown;
	double vel_y = (mVelocities[i].y + (real_t)force.y * velocity_update_fact) * this->mParams.slowdown;
	mVelocities[i].x = vel_x;
	mVelocities[i].y = vel_y;

	out_points[i].x = in_points[i].x + vel_x * this->mParams.timeQuantum;
	out_points[i].y = in_points[i].y + vel_y * this->mParams.timeQuantum;
}

/*
 * This is how a kernel call should be wrapped in a regular function call,
 * so it can be easilly used in cpp-only code.
 */
void run_my_kernel(float *src)
{
	my_kernel<<<64, 64>>>(src);
}

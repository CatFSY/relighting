#include "auxiliary.h"
#include <optix.h>

#include "gaussiantrace_forward.h"

namespace surfel_tracer {

extern "C" {
	__constant__ Gaussiantrace_forward::Params params;
}

template <bool FULL_OUTPUT, int MATERIAL_FEATURES>
static __forceinline__ __device__ void trace_gaussian_ray() {
	const uint3 idx = optixGetLaunchIndex();

	glm::vec3 ray_o = params.ray_origins[idx.x];
	glm::vec3 ray_d = params.ray_directions[idx.x];
	glm::vec3 ray_origin;

	glm::vec3 C = glm::vec3(0.0f, 0.0f, 0.0f), N = glm::vec3(0.0f, 0.0f, 0.0f);
	float D = 0.0f, O = 0.0f, T = 1.0f, t_start = 0.0f, t_curr = 0.0f;
	int trace_batches = 0, buffered_hits = 0;
	int alpha_hits = 0, full_hit_batches = 0;
	constexpr int FEATURE_STORAGE = FULL_OUTPUT ? MAX_FEATURE_SIZE :
		(MATERIAL_FEATURES > 0 ? MATERIAL_FEATURES : 1);
	float F[FEATURE_STORAGE] = {0.0f};

	HitInfo hitArray[MAX_BUFFER_SIZE];
	unsigned int hitArrayPtr0 = (unsigned int)((uintptr_t)(&hitArray) & 0xFFFFFFFF);
    unsigned int hitArrayPtr1 = (unsigned int)(((uintptr_t)(&hitArray) >> 32) & 0xFFFFFFFF);

	while ((t_start < T_SCENE_MAX) && (T > params.transmittance_min)){
		++trace_batches;
		ray_origin = ray_o + t_start * ray_d;
		
		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			hitArray[i].t = 1e16f;
			hitArray[i].primIdx = -1;
		}
		optixTrace(
			params.handle,
			make_float3(ray_origin.x, ray_origin.y, ray_origin.z),
			make_float3(ray_d.x, ray_d.y, ray_d.z),
			FLT_EPSILON,                // Min intersection distance
			T_SCENE_MAX,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
			0,                   // SBT offset
			1,                   // SBT stride
			0,                   // missSBTIndex
			hitArrayPtr0,
			hitArrayPtr1
		);
		if (hitArray[MAX_BUFFER_SIZE - 1].primIdx != -1) {
			++full_hit_batches;
		}

		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			int primIdx = hitArray[i].primIdx;

			if (primIdx == -1) {
				t_curr = T_SCENE_MAX;
				break;
			}
			else{
				++buffered_hits;
				t_curr = hitArray[i].t;
				int gs_idx = params.gs_idxs[primIdx];

				float o = params.opacity[gs_idx];
				glm::vec3 mean3D = params.means3D[gs_idx];
				glm::vec3 n = params.normals[gs_idx];
				glm::vec3 ru = params.ru[gs_idx];
				glm::vec3 rv = params.rv[gs_idx];

				float cos = -dot(ray_d, n);
				float multiplier = ((cos > 0)) ? 1: -1;
				if ((multiplier < 0) && params.back_culling) continue;
				glm::vec3 n_flip = multiplier * n;

				float o_g = glm::dot(n, (ray_o - mean3D));
				float d_g = glm::dot(n, ray_d);
				float d = -o_g * d_g / max(1e-6f, d_g * d_g);
				// Respect the original ray's parametric endpoint even after
				// candidate traversal advances ray_origin in multiple batches.
				// This is required for finite point/area-light shadow segments.
				if (d > T_SCENE_MAX) continue;
				glm::vec3 pos = ray_o + d * ray_d - mean3D;
				glm::vec2 p_g = {glm::dot(ru, pos), glm::dot(rv, pos)}; 
				float alpha = min(0.99, o * __expf(-0.5f * glm::dot(p_g, p_g)));

				if (alpha<params.alpha_min) continue;
				++alpha_hits;

				float w = T * alpha;
				if constexpr (FULL_OUTPUT) {
					glm::vec3 c = computeColorFromSH_forward(
						params.deg, ray_d,
						params.shs + gs_idx * params.max_coeffs);
					C += w * c;
					D += w * d;
				}
				if constexpr (FULL_OUTPUT || MATERIAL_FEATURES > 0) {
					N += w * n_flip;
				}
				O += w;

				if constexpr (FULL_OUTPUT) {
					for (int j = 0; j < params.S; ++j){
						F[j] += w * params.features[gs_idx * params.S + j];
					}
				} else if constexpr (MATERIAL_FEATURES > 0) {
					#pragma unroll
					for (int j = 0; j < MATERIAL_FEATURES; ++j){
						F[j] += w * params.features[
							gs_idx * MATERIAL_FEATURES + j];
					}
				}

				T *= (1 - alpha);

				if (T < params.transmittance_min){
					break;
				}
			}
			

		}
		t_start += t_curr;
	}
	
	if constexpr (FULL_OUTPUT) {
		params.color[idx.x] = C;
		params.depth[idx.x] = D;
	}
	if constexpr (FULL_OUTPUT || MATERIAL_FEATURES > 0) {
		params.normal[idx.x] = N;
	}
	params.alpha[idx.x] = O;
	if constexpr (FULL_OUTPUT) {
		for (int i = 0; i < params.S; ++i){
			params.feature[idx.x * params.S + i] = F[i];
		}
	} else if constexpr (MATERIAL_FEATURES > 0) {
		#pragma unroll
		for (int i = 0; i < MATERIAL_FEATURES; ++i){
			params.feature[idx.x * MATERIAL_FEATURES + i] = F[i];
		}
	}
	if (params.diagnostics_enabled) {
		int* diagnostic = params.diagnostics + idx.x * 4;
		diagnostic[0] = trace_batches;
		diagnostic[1] = buffered_hits;
		diagnostic[2] = alpha_hits;
		diagnostic[3] = full_hit_batches;
	}
}

extern "C" __global__ void __raygen__rg() {
	trace_gaussian_ray<true, 0>();
}

extern "C" __global__ void __raygen__material4() {
	trace_gaussian_ray<false, 4>();
}

extern "C" __global__ void __raygen__material5() {
	trace_gaussian_ray<false, 5>();
}

extern "C" __global__ void __raygen__visibility() {
	trace_gaussian_ray<false, 0>();
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ch() {
}

extern "C" __global__ void __anyhit__ah() {
    HitInfo* hitArray = (HitInfo*)((uintptr_t)optixGetPayload_0() | ((uintptr_t)optixGetPayload_1() << 32));

	float THit = optixGetRayTmax();
    int i_prim = optixGetPrimitiveIndex();
	HitInfo newHit = {THit, i_prim};

	for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
		if (hitArray[i].primIdx == -1){
			hitArray[i] = newHit;
			break;
		}
        else if (hitArray[i].t > newHit.t) {
			host_device_swap<HitInfo>(hitArray[i], newHit);
        }
    }

	if (THit < hitArray[MAX_BUFFER_SIZE - 1].t) {
        optixIgnoreIntersection(); 
    }

}

}

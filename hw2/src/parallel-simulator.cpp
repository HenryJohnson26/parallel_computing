#include "make_unique.h"
#include "world.h"
#include "quad-tree.h"
#include <algorithm>
#include <iostream>
#include <omp.h>

// TASK 2

// NOTE: You may modify this class definition as you see fit, as long as the class name,
// and type of simulateStep and buildAccelerationStructure remain the same.

const int QuadTreeLeafSize = 8;
const int QuadTreeParallelThreshold = 10000;
class ParallelNBodySimulator : public INBodySimulator
{
public:
    // TODO: implement a function that builds and returns a quadtree containing particles.
    // You do not have to preserve this function type.
    std::shared_ptr<QuadTreeNode> buildQuadTree(std::vector<Particle> & particles, Vec2 bmin, Vec2 bmax)
    {
        if (particles.size()<QuadTreeLeafSize){
        // return leaf node
        auto leaf = std::make_shared<QuadTreeNode>();
        leaf->isLeaf = true;
        leaf->particles = particles;
        return leaf;
       }
       else{
        auto nonleaf = std::make_shared<QuadTreeNode>();
        Vec2 pivot = (bmin + bmax) * 0.5f;
        std::vector<Particle> q0, q1, q2, q3;
        for (auto & p : particles) {
            bool right = p.position.x >= pivot.x;
            bool below = p.position.y >= pivot.y;
            if (!right && !below) q0.push_back(p); // top-left
            else if ( right && !below) q1.push_back(p); // top-right
            else if (!right &&  below) q2.push_back(p); // bottom-left
            else q3.push_back(p); // bottom-right
        }

        nonleaf->children[0] = buildQuadTree(q0, bmin, pivot);
        nonleaf->children[1] = buildQuadTree(q1, Vec2(pivot.x, bmin.y), Vec2(bmax.x, pivot.y));
        nonleaf->children[2] = buildQuadTree(q2, Vec2(bmin.x, pivot.y), Vec2(pivot.x, bmax.y));
        nonleaf->children[3] = buildQuadTree(q3, pivot, bmax);

        return nonleaf;
       }
    }

    // Do not modify this function type.
    virtual std::shared_ptr<AccelerationStructure> buildAccelerationStructure(std::vector<Particle> & particles)
    {
        // build quad-tree
        auto quadTree = std::make_shared<QuadTree>();

        // find bounds
        Vec2 bmin(1e30f, 1e30f);
        Vec2 bmax(-1e30f, -1e30f);

        for (auto & p : particles)
        {
            bmin.x = fminf(bmin.x, p.position.x);
            bmin.y = fminf(bmin.y, p.position.y);
            bmax.x = fmaxf(bmax.x, p.position.x);
            bmax.y = fmaxf(bmax.y, p.position.y);
        }

        quadTree->bmin = bmin;
        quadTree->bmax = bmax;
        // build nodes
        quadTree->root = buildQuadTree(particles, bmin, bmax);
        if (!quadTree->checkTree()) {
          std::cout << "Your Tree has Error!" << std::endl;
        }

        return quadTree;
    }

    // Do not modify this function type.
    virtual void simulateStep(AccelerationStructure * accel,
                            std::vector<Particle> & particles,
                            std::vector<Particle> & newParticles,
                            StepParameters params) override
    {
        // TODO: implement parallel version of quad-tree accelerated n-body simulation here,
        // using quadTree as acceleration structure
        auto qtree = static_cast<QuadTree*>(accel);
        int n = particles.size();
        int threads = omp_get_max_threads();
        if (n < 10000){
            #pragma omp parallel num_threads(8)
            {
                std::vector<Particle> local_ps;
                local_ps.reserve(64);
                #pragma omp for schedule(static)
                for(int i = 0; i < n; i++){
                    Particle& p = particles[i];
                    local_ps.clear();
                    qtree->getParticles(local_ps, p.position, params.cullRadius);
                    int n = local_ps.size();
                    {
                        Vec2 force(0.0f, 0.0f);
                        for (auto & other : local_ps){
                            if(other.id != p.id){
                                force = force + computeForce(p, other, params.cullRadius);
                            }
                        }
                        newParticles[i] = updateParticle(p, force, params.deltaTime);
                    }
                }
            }
        }
        else{
            #pragma omp parallel
            {
                std::vector<Particle> local_ps;
                local_ps.reserve(64);
                #pragma omp for schedule(dynamic, 64)
                for(int i = 0; i < n; i++){
                    Particle& p = particles[i];
                    local_ps.clear();
                    qtree->getParticles(local_ps, p.position, params.cullRadius);
                    int n = local_ps.size();
                    {
                        Vec2 force(0.0f, 0.0f);
                        for (auto & other : local_ps){
                            if(other.id != p.id){
                                force = force + computeForce(p, other, params.cullRadius);
                            }
                        }
                        newParticles[i] = updateParticle(p, force, params.deltaTime);
                    }
                }
            }
        }


    }
};

// Do not modify this function type.
std::unique_ptr<INBodySimulator> createParallelNBodySimulator()
{
  return std::make_unique<ParallelNBodySimulator>();
}
#pragma once


struct point
{
    float x, y, z;
};

struct camera_intrinsics
{
    explicit camera_intrinsics(const int w,
                               const int h,
                               const float fx,
                               const float fy,
                               const float cx,
                               const float cy,
                               const float k1 = 0.f,
                               const float k2 = 0.f,
                               const float k3 = 0.f,
                               const float p1 = 0.f,
                               const float p2 = 0.f) :
        h(h),
        w(w),
        fx(fx),
        fy(fy),
        cx(cx),
        cy(cy),
        k1(k1),
        k2(k2),
        k3(k3),
        p1(p1),
        p2(p2)
    {}

    int h, w;
    float fx, fy;
    float cx, cy;
    float k1, k2, k3;
    float p1, p2;
};

//+
SetFactory("OpenCASCADE");
Circle(1) = {-0, 0, 0, 0.5, 0, 2*Pi};

// Far-Field
lc_e = 10;
Point(2) = {-50, -50, 0.0, lc_e};
Point(3) = {50, -50, 0.0, lc_e};
Point(4) = {50, 50, 0.0, lc_e};
Point(5) = {-50, 50, 0.0, lc_e};

Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 2};

Curve Loop(1) = {1};
Curve Loop(2) = {2, 3, 4, 5}; 

Plane Surface(1) = {2, 1};

Mesh.MeshSizeFromCurvature = 250;
Mesh 2;
//+
Physical Surface("Domain", 6) = {1};
//+
Physical Curve("Topbot", 7) = {4, 2};
//+
Physical Curve("Inflow", 8) = {5};
//+
Physical Curve("Outflow", 9) = {3};
//+
Physical Curve("Wall", 10) = {1};

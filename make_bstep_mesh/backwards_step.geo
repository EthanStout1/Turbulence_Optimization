// Define Geometry
Ls = 6.0; // Step length
Ld = 20.0; // Domain length
H = 1.0; // Step height


// Backwards Step Points
lc_e = 0.07;
Point(1) = {0.0, 0.0, 0.0, lc_e};
Point(2) = {Ls, 0.0, 0.0, lc_e};
Point(3) = {Ls, -H, 0.0, lc_e};
Point(4) = {Ls + Ld, -H, 0.0, lc_e};
Point(5) = {Ls + Ld, H, 0.0, lc_e};
Point(6) = {0.0, H, 0.0, lc_e};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6}; 

Plane Surface(1) = {1};

//Mesh.MeshSizeFromCurvature = 250;
Mesh 2;
//+
Physical Surface("Domain", 7) = {1};
//+
Physical Curve("Inlet", 8) = {6};
//+
Physical Curve("Outlet", 9) = {4};
//+
Physical Curve("Walls_other", 10) = {5, 1, 2};
//+
Physical Curve("Wall_bot", 11) = {3};

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <Eigen/Dense>

const double f = 100.5;
const double k0 = 4;
const double k1 =17;
int max_iterations = 10;

struct Outlier {
    int index;       // 异常点索引
    double magnitude; // 对应的异常值
};

struct point2d
{
    int id;
    double x;
    double y;
};


// Function to calculate rotation matrices R1 and R2
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> Rotate(const Eigen::VectorXd &X) {
    double tao1 = X(0), k1 = X(1), e = X(2), tao2 = X(3), k2 = X(4);
    
    Eigen::Matrix3d R1, R2;

    R1 << std::cos(tao1) * std::cos(k1), -std::cos(tao1) * std::sin(k1), -std::sin(tao1),
          std::sin(k1), std::cos(k1), 0,
          std::sin(tao1) * std::cos(k1), -std::sin(tao1) * std::sin(k1), std::cos(tao1);
    
    R2 << std::cos(tao2) * std::cos(k2), -std::cos(tao2) * std::sin(k2), -std::sin(tao2),
          std::cos(e) * std::sin(k2) - std::sin(e) * std::sin(tao2) * std::cos(k2), 
          std::cos(e) * std::cos(k2) + std::sin(e) * std::sin(tao2) * std::sin(k2), 
          -std::sin(e) * std::cos(tao2),
          std::sin(e) * std::sin(k2) + std::cos(e) * std::sin(tao2) * std::cos(k2),
          std::sin(e) * std::cos(k2) - std::cos(e) * std::sin(tao2) * std::sin(k2),
          std::cos(e) * std::cos(tao2);

    return {R1, R2};
}

// Function to convert plane coordinates to space coordinates
void Planetospace(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2, const Eigen::MatrixXd &dataarr,
                  std::vector<Eigen::Vector3d> &a1, std::vector<Eigen::Vector3d> &a2) {
    for (int i = 0; i < dataarr.rows(); ++i) {
        Eigen::Vector3d xyz1(dataarr(i, 1), dataarr(i, 2), -f);
        Eigen::Vector3d xyz2(dataarr(i, 3), dataarr(i, 4), -f);
        a1.push_back(R1 * xyz1);
        a2.push_back(R2 * xyz2);
    }
}

// Function to compute error equation components
void Equationoferror(const std::vector<Eigen::Vector3d> &a1, const std::vector<Eigen::Vector3d> &a2,
                     const Eigen::VectorXd &X, Eigen::MatrixXd &A, Eigen::VectorXd &l, double &B) {
    std::vector<double> baseline_diffs;
    B=0;
    for (size_t i = 0; i < a1.size(); ++i) {
        B+=a1[i][0] - a2[i][0];
    }
    B =B/a1.size();
    //B=1;
    //B=1.0;
    A.resize(a1.size(), 5);
    l.resize(a1.size());

    for (size_t j = 0; j < a1.size(); ++j) {
        double x1 = a1[j][0], y1 = a1[j][1], z1 = a1[j][2];
        double x2 = a2[j][0], y2 = a2[j][1], z2 = a2[j][2];
        double tao1 = X(0), e = X(2), tao2 = X(3);

        A(j, 0) = (B * x1 * y2) / (x1 * z2 - x2 * z1);
        A(j, 1) = ((x1 * std::cos(tao1) + z1 * std::sin(tao1)) * B * z2 + B * y1 * y2 * std::sin(tao1)) / (x2 * z1 - x1 * z2);
        A(j, 2) = (B * y1 * y2 + B * z1 * z2) / (x2 * z1 - x1 * z2);
        A(j, 3) = (B * y1 * x2 * std::cos(e) + B * x2 * z1 * std::sin(e)) / (x2 * z1 - x1 * z2);
        A(j, 4) = ((y1 * x2 * std::sin(e) * std::cos(tao2) - y1 * y2 * std::sin(tao2) - z1 * x2 * std::cos(e) * std::cos(tao2) + z1 * z2 * std::sin(tao2)) * B) / (x2 * z1 - x1 * z2);

        double T = (B * z2) / (x1 * z2 - x2 * z1);
        double T_ = (B * z1) / (x1 * z2 - x2 * z1);
        l(j) = T * y1 - T_ * y2;
        // A(j, 0) = -(x1*y2)/z1;
        // A(j, 1) = -x1;
        // A(j, 2) = f+f*y1*y2/(z1*z2);
        // A(j, 3) = x2*y1/z1;
        // A(j, 4) = x2;

        // double T = (B * z2) / (x1 * z2 - x2 * z1);
        // double T_ = (B * z1) / (x1 * z2 - x2 * z1);
        // l(j) = f*y1/z1-f*y2/z2;
    }
}

std::vector<Outlier> processData(const Eigen::MatrixXd &dataarr) {
    // Initialize parameters
    Eigen::VectorXd X(5);
    X.setZero();

    // Calculate initial R1, R2
    auto [R1, R2] = Rotate(X);

    // Space coordinates
    std::vector<Eigen::Vector3d> a1, a2;
    Planetospace(R1, R2, dataarr, a1, a2);

    // Initialize A, l, B and calculate initial values
    Eigen::MatrixXd A;
    Eigen::VectorXd l;
    double B;
    Equationoferror(a1, a2, X, A, l, B);
    // Define remaining components
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(l.size(), l.size());
    Eigen::VectorXd W = A.transpose() * P * l;
    Eigen::MatrixXd N = A.transpose() * P * A;
    Eigen::MatrixXd Qxx = N.inverse();
    Eigen::VectorXd deltaX = Qxx * W;
    Eigen::VectorXd V = A * deltaX - l;
    Eigen::MatrixXd Qvv = P.inverse() - A * Qxx * A.transpose();
    Eigen::VectorXd P_new = P.diagonal();
    double sigma0 = std::sqrt((V.transpose() * P * V).value() / (l.size() - 5));
    X += deltaX;
    // Store outliers
    std::vector<Outlier> outliers;
    //std::cout<<std::abs(a1[0][1]-a2[0][1])<<std::endl;
    // Iterative outlier detection loop
    for (int m = 0; m < max_iterations; ++m) {
        auto [R1, R2] = Rotate(X);
        std::vector<Eigen::Vector3d> a1, a2;
        Planetospace(R1, R2, dataarr, a1, a2);
        Eigen::MatrixXd A;
        Eigen::VectorXd l;
        double B;
        Equationoferror(a1, a2, X, A, l, B);
       
        //P_new = P.diagonal();
        // Clear previous outliers
        outliers.clear();

        for (int i = 0; i < V.size(); ++i) {
            //double v_ = V(i) / (sigma0 * std::sqrt(Qvv(i, i)));
            //double v_ = V(i) / (0.0028 * std::sqrt(Qvv(i, i)));
            //double v_ = std::sqrt(Qvv(i, i));
            double v_ = V(i) / (0.0028);
            //double v_=std::abs(a1[i][1]-a2[i][1])/(0.0028);
            // Add to outliers if it exceeds threshold
            if (std::abs(v_) > k1) {
                outliers.push_back({int(dataarr(i,0)), std::abs(v_)});
            }
            // if (std::abs(a1[i][1]-a2[i][1]) > 4*0.0028) {
            //     //outliers.push_back({int(dataarr(i,0)), std::abs(a1[i][1]-a2[i][1])});
            //     outliers.push_back({int(dataarr(i,0)), std::abs(a1[i][1]-a2[i][1])});
            // }
            //std::cout<<std::abs(a1[i][1]-a2[i][1])<<std::endl;
            
            double k;
            // if(v_>3)
            // {
            //     k=exp(1-(v_/3.0)*(v_/3.0));
            // }
            if (std::abs(v_) < k0) {
                k = 1.0;
            } else if (std::abs(v_) > k1) {
                k = 1e-8;
            } else {
                k = (k0 / std::abs(v_)) * std::pow((k1 - std::abs(v_)) / (k1 - k0), 2);
            }

            P_new(i) = P(i, i)*k;
        }

        P.diagonal() = P_new;
        //std::cout<<P_new<<std::endl;
        W = A.transpose() * P * l;
        N = A.transpose() * P * A;
        Qxx = N.inverse();
        Qvv = P.inverse() - A * Qxx * A.transpose();
        deltaX = Qxx * W;
        V = A * deltaX - l;
        sigma0 = std::sqrt((V.transpose() * P * V).value() / (l.size() - 5));
        X += deltaX;
        
       std::cout<<deltaX(0)<<std::endl;
    }

    return outliers;
}



void readData(const std::string& filename, 
                                      Eigen::MatrixXd& dataarr1, 
                                      Eigen::MatrixXd& dataarr2,
                                       Eigen::MatrixXd& dataarr3) {
    std::ifstream file(filename);
    std::string line;
    std::vector<point2d> current_image_points;
    int current_image_id = -1;
    std::vector<point2d> image1_points;
    std::vector<point2d> image2_points; 
    std::vector<point2d> image3_points;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int id;
        double x, y;

        ss >> id >> x >> y;

        if (id == -99) {
            // -99 indicates end of one image's data, switch to next image
            if (current_image_id == 205021) {
                image1_points = current_image_points;
            } else if (current_image_id == 205022) {
                image2_points = current_image_points;
            } else if (current_image_id == -205023) {
                image3_points = current_image_points;
            }
            current_image_points.clear();
            continue;
        }

        if (id == 205021 || id == 205022 || id == -205023) {
            current_image_id = id;
        } else {
            // Otherwise, it's a point
            point2d pt = {id, x, y};
            current_image_points.push_back(pt);
        }
    }

    // Find common points between image1 and image2, store in dataarr1
    std::vector<point2d> common_points12;
    for (const auto& pt1 : image1_points) {
        for (const auto& pt2 : image2_points) {
            if (pt1.id == pt2.id) {
                common_points12.push_back(pt1);
                dataarr1.conservativeResize(dataarr1.rows() + 1, 5);
                dataarr1(dataarr1.rows() - 1, 0) = pt1.id;
                dataarr1(dataarr1.rows() - 1, 1) = pt1.x;
                dataarr1(dataarr1.rows() - 1, 2) = pt1.y;
                dataarr1(dataarr1.rows() - 1, 3) = pt2.x;
                dataarr1(dataarr1.rows() - 1, 4) = pt2.y;
            }
        }
    }

    // Find common points between image1 and image3, store in dataarr2
    std::vector<point2d> common_points13;
    for (const auto& pt1 : image1_points) {
        for (const auto& pt3 : image3_points) {
            if (pt1.id == pt3.id) {
                common_points13.push_back(pt1);
                dataarr2.conservativeResize(dataarr2.rows() + 1, 5);
                dataarr2(dataarr2.rows() - 1, 0) = pt1.id;
                dataarr2(dataarr2.rows() - 1, 1) = pt1.x;
                dataarr2(dataarr2.rows() - 1, 2) = pt1.y;
                dataarr2(dataarr2.rows() - 1, 3) = pt3.x;
                dataarr2(dataarr2.rows() - 1, 4) = pt3.y;
            }
        }
    }
    std::vector<point2d> common_points23;
    for (const auto& pt2 : image2_points) {
        for (const auto& pt3 : image3_points) {
            if (pt2.id == pt3.id) {
                common_points23.push_back(pt2);
                dataarr3.conservativeResize(dataarr3.rows() + 1, 5);
                dataarr3(dataarr3.rows() - 1, 0) = pt2.id;
                dataarr3(dataarr3.rows() - 1, 1) = pt2.x;
                dataarr3(dataarr3.rows() - 1, 2) = pt2.y;
                dataarr3(dataarr3.rows() - 1, 3) = pt3.x;
                dataarr3(dataarr3.rows() - 1, 4) = pt3.y;
            }
        }
    }
}
void find_outlier_intersection(const std::vector<Outlier>& outliers1,
                               const std::vector<Outlier>& outliers2,
                               std::vector<Outlier>& outliersresult) {
    outliersresult.clear(); // 确保结果容器为空

    // 遍历 outliers1 中的每个 Outlier
    for (const auto& o1 : outliers1) {
        // 在 outliers2 中查找具有相同 index 的 Outlier
        auto it = std::find_if(outliers2.begin(), outliers2.end(),
                               [&o1](const Outlier& o2) {
                                   return o1.index == o2.index;
                               });

        // 如果找到了，加入结果
        if (it != outliers2.end()) {
            outliersresult.push_back(o1);
        }
    }
}
void find_outlier_union(const std::vector<Outlier>& outliers1,
                        const std::vector<Outlier>& outliers2,
                        std::vector<Outlier>& outliersresult) {
    outliersresult.clear(); // 确保结果容器为空

    // 首先添加 outliers1 中的所有元素到结果
    for (const auto& o1 : outliers1) {
        outliersresult.push_back(o1);
    }

    // 然后检查 outliers2 中的元素，如果其 id 不在结果中，则添加
    for (const auto& o2 : outliers2) {
        bool found = false;
        for (const auto& existing : outliersresult) {
            if (existing.index == o2.index) {
                found = true;
                break;
            }
        }
        if (!found) {
            outliersresult.push_back(o2);
        }
    }
}

int main() {

    Eigen::MatrixXd dataarr1;Eigen::MatrixXd dataarr2;Eigen::MatrixXd dataarr3;
    readData("PREPHIy10.DAT",dataarr1,dataarr2,dataarr3);
    //std::cout<<dataarr1<<std::endl;
    //std::cout<<dataarr2<< std::endl;
    std::vector<Outlier> outliers1 = processData(dataarr1);
    std::vector<Outlier> outliers2 = processData(dataarr2);
    std::vector<Outlier> outliers3 = processData(dataarr3);
    std::vector<Outlier> outliersresult1;
    std::vector<Outlier> outliersresult2;
    std::vector<Outlier> outliersresult;
    // for (const auto &outlier : outliers1) {
    //     std::cout << "Index: " << outlier.index
    //               << ", Magnitude: " << outlier.magnitude << std::endl;
    // }
    // for (const auto &outlier : outliers2) {
    //     std::cout << "Index: " << outlier.index
    //               << ", Magnitude: " << outlier.magnitude << std::endl;
    // }
    
    find_outlier_union(outliers1,outliers2,outliersresult1);
    find_outlier_union(outliers3,outliersresult1,outliersresult2);
    //find_outlier_intersection(outliers3,outliers2,outliersresult2);
    //find_outlier_intersection(outliersresult1,outliersresult2,outliersresult1);
    std::cout<<"number: "<<outliersresult1.size()<<std::endl;
    for (const auto &outlier : outliersresult1) {
        std::cout << "Index: " << outlier.index
                  << ", Magnitude: " << outlier.magnitude << std::endl;
    }

    return 0;
}

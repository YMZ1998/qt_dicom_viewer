import os
import sys
from collections import defaultdict

import numpy as np
import pydicom
from shapely.geometry import Polygon


# =========================
# ROI 数据结构
# =========================
class ROI:
    def __init__(self, roi_number, roi_name):
        self.number = roi_number
        self.name = roi_name
        # z(float) -> list[Polygon]
        self.slices = defaultdict(list)

    def add_polygon(self, z, xy):
        # 检查坐标点数量是否足够形成一个多边形
        if len(xy) < 3:
            return  # 至少需要3个点才能形成多边形
        elif len(xy) == 3:
            # 对于只有3个点的情况，重复第一个点以闭合多边形
            xy = xy + [xy[0]]

        try:
            poly = Polygon(xy)
            if poly.is_valid and poly.area > 0:
                self.slices[z].append(poly)
        except Exception as e:
            print(f"[WARNING] 创建多边形时出错: {e}, 坐标数: {len(xy)}")
            return


# =========================
# 读取 RTSTRUCT
# =========================
def load_rois_from_rtss(rtss_path, z_tol=1e-3, roi_names_filter=None):
    if not os.path.exists(rtss_path):
        raise FileNotFoundError(f"RTSTRUCT 文件不存在: {rtss_path}")
    else:
        print(f"[INFO] 正在读取 RTSTRUCT 文件: {rtss_path}")

    try:
        ds = pydicom.dcmread(rtss_path)
    except Exception as e:
        raise ValueError(f"无法读取 RTSTRUCT 文件: {e}")

    # 检查是否包含必需的序列
    if not hasattr(ds, 'StructureSetROISequence'):
        raise ValueError("RTSTRUCT 文件缺少 StructureSetROISequence")

    if not hasattr(ds, 'ROIContourSequence'):
        raise ValueError("RTSTRUCT 文件缺少 ROIContourSequence")

    # ROINumber -> ROIName
    roi_info = {
        roi.ROINumber: roi.ROIName
        for roi in ds.StructureSetROISequence
    }
    print(f"[INFO] 发现 ROI: {roi_info}")

    # 如果指定了ROI过滤器，则只处理指定的ROI
    if roi_names_filter:
        filtered_roi_numbers = []
        for num, name in roi_info.items():
            if name in roi_names_filter:
                filtered_roi_numbers.append(num)
        print(f"[INFO] 已应用ROI过滤器，仅处理: {roi_names_filter}")
        # 只初始化指定的ROI对象
        rois = {
            num: ROI(num, roi_info[num])
            for num in filtered_roi_numbers
            if num in roi_info
        }
    else:
        # 初始化所有ROI对象
        rois = {
            num: ROI(num, name)
            for num, name in roi_info.items()
        }

    # 解析 Contour
    for roi_contour in ds.ROIContourSequence:
        roi_number = roi_contour.ReferencedROINumber
        roi_obj = rois.get(roi_number)  # 只处理感兴趣的ROI
        if roi_obj is None:
            # if roi_names_filter:
            #     print(f"[INFO] 跳过未指定的 ROINumber {roi_number} ({roi_info.get(roi_number, 'Unknown')})")
            # else:
            #     print(f"[WARNING] 找不到 ROINumber {roi_number} 对应的 ROI 信息")
            continue

        if not hasattr(roi_contour, "ContourSequence"):
            print(f"[WARNING] ROINumber {roi_number} 缺少 ContourSequence")
            continue

        for contour in roi_contour.ContourSequence:
            if not hasattr(contour, 'ContourData'):
                print(f"[WARNING] 发现缺少 ContourData 的轮廓")
                continue

            try:
                data = np.array(contour.ContourData).reshape(-1, 3)
            except ValueError as e:
                print(f"[WARNING] 无法解析轮廓数据: {e}")
                continue

            # 处理 z 浮点误差
            z = float(np.mean(data[:, 2]))
            z = round(z / z_tol) * z_tol

            xy = [(float(x), float(y)) for x, y in data[:, :2]]
            roi_obj.add_polygon(z, xy)

    # 过滤掉没有切片的 ROI
    valid_rois = [roi for roi in rois.values() if len(roi.slices) > 0]
    print(f"[INFO] 成功加载 {len(valid_rois)} 个有效的 ROI")
    return valid_rois


# =========================
# ROI 是否相交
# =========================
def roi_intersect(roi_a, roi_b):
    common_slices = set(roi_a.slices.keys()) & set(roi_b.slices.keys())
    if not common_slices:
        return False

    for z in common_slices:
        for pa in roi_a.slices[z]:
            for pb in roi_b.slices[z]:
                # bbox 快速剪枝
                try:
                    minx1, miny1, maxx1, maxy1 = pa.bounds
                    minx2, miny2, maxx2, maxy2 = pb.bounds
                except AttributeError:
                    # 如果 bounds 属性不可用，则跳过
                    continue

                if maxx1 < minx2 or maxx2 < minx1:
                    continue
                if maxy1 < miny2 or maxy2 < miny1:
                    continue

                try:
                    intersection_area = pa.intersection(pb).area
                    if intersection_area > 0:
                        return True
                except Exception:
                    # 如果交集计算失败，尝试使用其他方法判断
                    try:
                        if pa.intersects(pb):
                            return True
                    except Exception:
                        continue
    return False


# =========================
# 构建冲突图
# =========================
def build_conflict_graph(rois):
    graph = {roi.name: set() for roi in rois}

    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            if roi_intersect(rois[i], rois[j]):
                a, b = rois[i].name, rois[j].name
                graph[a].add(b)
                graph[b].add(a)

    return graph


# =========================
# DSATUR 图着色
# =========================
def dsatur_coloring(graph):
    colors = {}
    saturation = {v: 0 for v in graph}
    degrees = {v: len(graph[v]) for v in graph}
    uncolored = set(graph.keys())

    while uncolored:
        # saturation 最大，tie-break 用 degree
        v = max(uncolored, key=lambda x: (saturation[x], degrees[x]))

        neighbor_colors = {colors[n] for n in graph[v] if n in colors}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[v] = color

        for n in graph[v]:
            if n in uncolored:
                used = {colors[k] for k in graph[n] if k in colors}
                saturation[n] = len(used)

        uncolored.remove(v)

    return colors


# =========================
# 颜色 → 集合
# =========================
def split_into_sets(colors):
    sets = defaultdict(list)
    for roi_name, color in colors.items():
        sets[color].append(roi_name)
    return sets


def save_results(roi_sets, output_path=None):
    """
    保存结果到文件或打印到控制台
    """
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"非重叠 ROI 分组结果 (共 {len(roi_sets)} 组)\n")
            f.write("=" * 50 + "\n")
            for idx in sorted(roi_sets.keys()):
                f.write(f"\n第 {idx} 组:\n")
                for name in roi_sets[idx]:
                    f.write(f"  - {name}\n")
        print(f"[INFO] 结果已保存到: {output_path}")
    else:
        print(f"\n[RESULT] 分割为 {len(roi_sets)} 个非重叠集合:\n")
        for idx in sorted(roi_sets.keys()):
            print(f"第 {idx} 组:")
            for name in roi_sets[idx]:
                print("  ", name)


def main():
    import argparse
    import ast

    # 定义默认路径
    # DEFAULT_RTSS_PATH = r"\\192.168.0.66\nas\test-data\28 训练数据\npc2\1\RS1.2.826.0.1.3680043.8.176.2020330151729119.81.3751426738.dcm"
    DEFAULT_RTSS_PATH = r"C:/Users/Admin/Desktop/20-1741_LIBOCHUAN/RTSTRUCT1.2.276.0.7230010.3.1.4.1092969453.6072.1594198040.375.dcm"
    # DEFAULT_RTSS_PATH = r"\\192.168.0.66\nas\test-data\28 训练数据\新华医院-自动勾画训练数据2\全部数据\17-2369_JIGUANQI\RTSTRUCT2.16.840.1.114362.1.6.7.5.17616.11429693373.469879045.546.3550.dcm"

    parser = argparse.ArgumentParser(description='将有重叠的ROI分割为非重叠的集合')
    parser.add_argument('--input_path', type=str, default=DEFAULT_RTSS_PATH,
                        help='输入的RTSTRUCT文件路径 (默认: {})'.format(DEFAULT_RTSS_PATH))
    parser.add_argument('-o', '--output', help='输出结果文件路径 (可选)')
    parser.add_argument('--z-tolerance', type=float, default=1e-3,
                        help='Z轴容差，默认1e-3')
    parser.add_argument('--rois', type=str, help='指定要处理的ROI名称列表，例如: "[\"ROI1\", \"ROI2\"]"')
    args = parser.parse_args()

    # args.rois = "['GTV-nx', 'GTV-nd', 'CTV-1', 'CTV-2', 'PTV-nx', 'PTV-nd', 'PTV-1', 'PTV-2']"

    # 解析ROI名称列表
    roi_names_filter = None
    if args.rois:
        try:
            roi_names_filter = ast.literal_eval(args.rois)
            if not isinstance(roi_names_filter, list):
                raise ValueError("ROI参数必须是一个列表")
            print(f"[INFO] 将处理指定的ROI: {roi_names_filter}")
        except (ValueError, SyntaxError) as e:
            print(f"[ERROR] ROI参数格式错误: {e}")
            print("示例格式: --rois '[\"ROI1\", \"ROI2\"]'")
            sys.exit(1)

    try:
        print(f"[INFO] 开始处理 RTSTRUCT 文件: {args.input_path}")
        rois = load_rois_from_rtss(args.input_path, z_tol=args.z_tolerance, roi_names_filter=roi_names_filter)
        print(f"[INFO] 成功加载 {len(rois)} 个 ROI")

        if len(rois) == 0:
            print("[ERROR] 没有找到任何有效的 ROI")
            sys.exit(1)

        print("[INFO] 构建冲突图...")
        graph = build_conflict_graph(rois)

        print("[INFO] 应用DSATUR算法进行图着色...")
        colors = dsatur_coloring(graph)

        print("[INFO] 生成分组结果...")
        roi_sets = split_into_sets(colors)

        save_results(roi_sets, args.output)

        print(f"\n[SUMMARY] 处理完成!")
        print(f"  - 总ROI数: {sum(len(roi_set) for roi_set in roi_sets.values())}")
        print(f"  - 分组数: {len(roi_sets)}")
        for idx in sorted(roi_sets.keys()):
            print(f"  - 第 {idx} 组: {len(roi_sets[idx])} 个 ROI")

    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] 数据错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 处理过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

def font_get():
    """
    加载Times New Roman和宋体(SimSun)字体，确保Matplotlib正常显示中英文
    - Times New Roman路径: /usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf
    - 宋体路径: /usr/share/fonts/truetype/simsun.ttc
    """
    import os
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # 定义字体路径
    tnr_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    simsun_font_path = '/usr/share/fonts/truetype/simsun.ttc'

    # 加载Times New Roman字体
    if os.path.exists(tnr_font_path):
        fm.fontManager.addfont(tnr_font_path)
        print(f"成功加载Times New Roman字体: {tnr_font_path}")
    else:
        print(f"警告：Times New Roman字体文件不存在: {tnr_font_path}")

    # 加载宋体(SimSun)字体
    if os.path.exists(simsun_font_path):
        fm.fontManager.addfont(simsun_font_path)
        print(f"成功加载宋体(SimSun)字体: {simsun_font_path}")
    else:
        print(f"警告：宋体字体文件不存在: {simsun_font_path}")

    # 方案2（可选）：优先Times New Roman（适合英文为主的场景）
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'],
        'font.sans-serif': ['Times New Roman', 'SimSun'],
        'axes.unicode_minus': False,
        'font.size': 12
    })

    # 验证字体加载结果
    try:
        # 检查Times New Roman
        tnr_fp = fm.FontProperties(family='Times New Roman')
        tnr_loaded = 'Times_New_Roman' in fm.findfont(tnr_fp)
        # 检查宋体
        simsun_fp = fm.FontProperties(family='SimSun')
        simsun_loaded = 'simsun' in fm.findfont(simsun_fp).lower()
        
        print(f"\n字体加载验证：")
        print(f"- Times New Roman: {'✓ 成功' if tnr_loaded else '✗ 失败'}")
        print(f"- 宋体(SimSun): {'✓ 成功' if simsun_loaded else '✗ 失败'}")
    except Exception as e:
        print(f"字体验证失败: {e}")
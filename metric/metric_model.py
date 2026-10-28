from metric.test_metric import evaluation_metric

task = ['SOD', 'RGB-D']
task = ['RGB-D']
task = ['RGB-T']

for t in task:
    print(f"\n🎯 Starting evaluation for task: {t}")

    if t == 'SOD':
        test_roots = ['DUT-OMRON', 'DUTS/Test', 'CSOD10K/test', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    elif t == 'RGB-D':
        #test_roots = ['DUT-RGBD/Test', 'NJU2K', 'NLPR', 'STERE', 'SIP']
        test_roots = ['DUT-RGBD', 'NJU2K', 'NLPR', 'SIP', 'STERE']
    elif t == 'RGB-T':
        test_roots = ['VT5000/Test', 'VT1000', 'VT821']
        #test_roots = ['unaligned_dataset/VT821_unalign', 'unaligned_dataset/VT1000_unalign', 'unaligned_dataset/VT5000-Test_unalign']

    for r in test_roots:
        print(f"\n🚀 Testing on dataset: {r}")
    for r in test_roots:
      pred_path = "./pred/" + r
      #pred_path = "./experiment_pred/pred_rgbd_full/" + r
      gt_root = './datasets/' + t + '/' + r + '/GT/'
      results = evaluation_metric(gt_root, pred_path)
      
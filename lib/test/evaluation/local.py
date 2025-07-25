from .environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/jzuser/Work_dir/SeqTrackv2/lib/test/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.our_data_path = '/home/jzuser/Work_dir/SeqTrackv2/data/our_data'
    settings.result_plot_path = '/home/jzuser/Work_dir/SeqTrackv2/lib/test/result_plots/'
    settings.results_path = '/home/jzuser/Work_dir/SeqTrackv2/lib/test/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/jzuser/Work_dir/SeqTrackv2/lib/test/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.prj_dir = '/home/jzuser/Work_dir/SeqTrackv2/'
    settings.save_dir = '/home/jzuser/Work_dir/SeqTrackv2/'  # Where to save evaluation results

    return settings


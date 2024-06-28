python 0_construct_yp_extend_features.py --root train
python 0_construct_yp_extend_features.py --root validation
python 0_construct_yp_extend_features.py --root test
python 1_construct_category_and_subcategory_features.py --root train
python 1_construct_category_and_subcategory_features.py --root validation
python 1_construct_category_and_subcategory_features.py --root test

python 1.0generate_article_features.py 
python 2.0generate_each_embedding.py 
python 3.0generate_explode_data.py 
python 4.0gen_bpr_feature.py 
python 5.1deal_article_data.py 
python 5.2deal_user_his_data.py 
python 5.3train_w2v.py 
python 5.4gen_sim_feature.py 
python 6.0cal_article_publish_48hours_feature.py 
python 7.0cal_his_inviews_cate_dedup_cnt.py 
python 8.0merge_feature_v1.py 
python 9.0merge_feature_v2.py 
python 10.0merge_feature_v3.py 
python 11.0gen_session_feature_and_merge_feature_v4.py 
python 12.0add_day_hour_article_pv_feature.py 
python 13.0add_new_feature_0614.py 
python 14.0add_new_feature_0619.py 
python 15.0add_new_feature_0620.py
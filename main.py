from main_helper import set_seed, extract_short_passing_play_data, generate_input_tensors, create_withheld_testing_set, cross_validation_cnn, cross_validation_logreg, baseline_model, train_final_cnn, evaluate_test_set

def main():
    print('running main')
    set_seed()

    # run_EXTRACT_DATA = True
    # run_GENERATE_TENSORS = False
    # run_CREATE_WITHHELD_TEST_SET = True

    run_CROSS_VAL_CNN = False
    run_CROSS_VAL_LOGREG = False
    run_BASELINE = False
    run_TRAIN_FINAL_CNN = True

    run_EVALUATE_TEST_SET = False

    # CODE TO EXTRACT THE RELEVANT SHORT PLAYS AND BUILD THE INPUT TENSORS FOR THE MODELS
    # CURRENTLY BUGGY, SO JUST USE THE ALREADY SAVED TENSORS (IN "model_inputs" FOLDER)
    ######################################################################################################

    # if run_EXTRACT_DATA:
    #     play_data_2021, tracking_data_2021, play_data_2022, tracking_data_2022 = extract_short_passing_play_data()
    
    # if run_GENERATE_TENSORS:
    #     generate_input_tensors()

    # if run_CREATE_WITHHELD_TEST_SET and play_data_2021:
    #     create_withheld_testing_set(play_data_2021, tracking_data_2021, play_data_2022, tracking_data_2022)
    # else:
    #     play_data_2021, tracking_data_2021, play_data_2022, tracking_data_2022 = extract_short_passing_play_data()
    #     create_withheld_testing_set(play_data_2021, tracking_data_2021, play_data_2022, tracking_data_2022)

    # VALIDATION
    ######################################################################################################
    if run_CROSS_VAL_CNN:
        cross_validation_cnn()

    if run_CROSS_VAL_LOGREG:
        cross_validation_logreg()

    if run_BASELINE:
        baseline_model()
    
    # TRAIN FINAL MODEL
    ######################################################################################################
    if run_TRAIN_FINAL_CNN:
        train_final_cnn()

    # GENERATE SPSP PROBS FOR ALL WITHHELD PLAYS (FROM BALL_SNAP TO PASS_FORWARD), AND GENERATE VISUALS
    ######################################################################################################
    if run_EVALUATE_TEST_SET:
        evaluate_test_set()



if __name__ == "__main__":
    main()
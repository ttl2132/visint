#user = [1,1,1,0,0]
#array = [1,1,0,0,1]
#testImages = ["front_tian.jpg", "newtest.jpg", "Fed.jpg", "tian_side_eye.dng", "tian_closed_eye.dng", "side_tian.jpg", "bry.jpg"]

#user = [1, 1, 1, 1, 0, 0, 1]
#array = [] #whatever is outputted in positionTracker

def user_test(user, array):
    #amount person is deemed to be focused
    focused = 0
    unfocused = 0
    for i in array:
        if i == 1:
            focused = focused + 1
        else:
            unfocused = unfocused + 1
    amount_focused = (focused/len(array))*100
    print("user is focused ", amount_focused, "% of the time")

    #comparing accuracy of program output with "correct" output
    output = []
    j = 0
    for i in user:
        if i == array[j]:
            output.append(1)
        else:
            output.append(0)
        j = j + 1

    count = 0
    for i in output:
        if i == 1:
            count = count+1
        else:
            count = count
    length = len(output)
    accuracy = count/length
    print("accuracy of program output: ", accuracy)

#if __name__ == "__main__":
#    user_test(user,array)
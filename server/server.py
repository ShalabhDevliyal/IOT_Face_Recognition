from flask import Flask, request, jsonify
import util 

@app.route("/classify_image", methods=['GET','POST'])
def classify_image():
    image_data = request.form['image_data']

    jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting the server.....")
    util.load_saved_artifacts()
    return app.run(debug=True)
use anyhow::{anyhow, Result};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use once_cell::sync::Lazy;
use polars::{datatypes::AnyValue, frame::DataFrame, prelude::NamedFrom, series::Series};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    env::{self, current_dir},
    sync::{Arc, Mutex},
};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

/// Represents the possible movements of the drone.
///
/// The `Movements` enum defines the different types of movements that the drone can perform.
/// Each variant corresponds to a specific movement.
///
/// # Variants
///
/// * `Takeoff` - Represents the takeoff movement.
/// * `Right` - Represents the right movement.
/// * `Left` - Represents the left movement.
/// * `Land` - Represents the land movement.
/// * `Forward` - Represents the forward movement.
/// * `Backward` - Represents the backward movement.
#[derive(Debug, Copy, Clone)]
enum Movements {
    Takeoff,
    Right,
    Left,
    Land,
    Forward,
    Backward,
}

/// Implements the `TryFrom` trait for converting a `usize` to a `Movements` enum variant.
///
/// This implementation allows converting a `usize` value to the corresponding `Movements` variant.
/// If the `usize` value is invalid (i.e., outside the range of valid variants), an error is returned.
impl TryFrom<usize> for Movements {
    type Error = &'static str;
    fn try_from(value: usize) -> std::prelude::v1::Result<Self, Self::Error> {
        match value {
            0 => Ok(Movements::Takeoff),
            1 => Ok(Movements::Right),
            2 => Ok(Movements::Left),
            3 => Ok(Movements::Land),
            4 => Ok(Movements::Forward),
            5 => Ok(Movements::Backward),
            _ => Err("Invalid usize"),
        }
    }
}

/// Implements the `Into` trait for converting a `Movements` enum variant to a `String`.
///
/// This implementation allows converting a `Movements` variant to its corresponding string representation.
/// The string representation is a lowercase version of the variant name.
impl Into<String> for Movements {
    fn into(self) -> String {
        match self {
            Movements::Takeoff => String::from("takeoff"),
            Movements::Right => String::from("right"),
            Movements::Left => String::from("left"),
            Movements::Land => String::from("land"),
            Movements::Forward => String::from("forward"),
            Movements::Backward => String::from("backward"),
        }
    }
}

/// Represents the shared state of the application.
///
/// The `AppState` struct contains the TensorFlow session, graph, and other shared data
/// that is accessible across multiple requests. It uses `Arc` and `Mutex` to allow
/// concurrent access to the shared data.
///
/// # Fields
///
/// * `session` - The TensorFlow session used for running predictions.
/// * `graph` - The TensorFlow graph representing the loaded model.
/// * `perdictions` - A vector of previous predictions stored as strings.
/// * `input_op` - The name of the input operation in the TensorFlow graph.
/// * `output_op` - The name of the output operation in the TensorFlow graph.
#[derive(Clone, Debug)]
struct AppState {
    session: Arc<SavedModelBundle>,
    graph: Arc<Graph>,
    perdictions: Arc<Mutex<Vec<String>>>,
}

/// Represents the response structure for a prediction request.
///
/// The `PredictionResponse` struct is used to serialize the prediction result
/// and send it back as a JSON response to the client.
///
/// # Fields
///
/// * `prediction_label` - The predicted label as a string.
/// * `prediction_count` - The count of predictions made so far.
#[derive(Serialize, Deserialize)]
struct PredictionResponse {
    prediction_label: String,
    prediction_count: usize,
}

/// The default address and port for the server to listen on.
const ADDRESS: Lazy<String> =
    Lazy::new(|| env::var("PREDITION_SERVER_ADDRESS").unwrap_or("0.0.0.0:5000".into()));

/// The default path to the TensorFlow SavedModel directory.
const MODEL: Lazy<String> = Lazy::new(|| env::var("PREDICTION_MODEL").unwrap_or("model.pb".into()));

/// The default name of the input operation in the TensorFlow graph.
const INPUT_OPERATION_NAME: Lazy<String> = Lazy::new(|| {
    env::var("PREDICTION_INPUT_OPERATION_NAME").unwrap_or("serving_default_input".into())
});

/// The default name of the output operation in the TensorFlow graph.
const OUTPUT_OPERATION_NAME: Lazy<String> = Lazy::new(|| {
    env::var("PREDICTION_OUTPUT_OPERATION_NAME").unwrap_or("StatefulPartitionedCall".into())
});

/// # Environment Variables
///
/// * `PREDICTION_MODEL` - The file path to the TensorFlow SavedModel directory.
///   Default value: `MODEL`.
/// * `PREDITION_SERVER_ADDRESS` - The address and port on which the server will listen.
///   Default value: `ADDR`.
/// * `PREDICTION_INPUT_OPERATION_NAME` - The name of the input operation in the TensorFlow graph.
///   Default value: `INPUT_OPERATION_NAME`.
/// * `PREDICTION_OUTPUT_OPERATION_NAME` - The name of the output operation in the TensorFlow graph.
///   Default value: `OUTPUT_OPERATION_NAME`.
///
#[tokio::main]
async fn main() -> Result<()> {
    // Read the configuration values from environment variables, or use default values
    // env::set_var("TF_CPP_MIN_LOG_LEVEL", "2"); // Ignore warning messages

    // Import the TensorFlow model and get the session and graph
    let (session, graph) = import_tensor(MODEL.to_string())?;

    // Create application state
    let state = AppState {
        session: Arc::new(session),
        graph: Arc::new(graph),
        perdictions: Arc::new(Mutex::new(vec![])),
    };

    // Create the application routes
    let app = Router::new()
        .route("/eegrandomforestprediction", post(post_predict))
        .route("/prediction", post(egg_prediction))
        .route("/time_prediction", post(egg_timestamp_prediction))
        .route("/lastpredition", get(last_prediction))
        .route("/takeoff", post(takeoff))
        .route("/land", post(land))
        .route("/forward", post(forward))
        .route("/backward", post(backward))
        .route("/left", post(left))
        .route("/rigth", post(right))
        .route("/imateapot", get(im_a_teapot))
        .with_state(state);

    // Create a TCP listener and bind it to the specified address
    let listener = tokio::net::TcpListener::bind(ADDRESS.to_string()).await?;
    println!("listening on http://{}", listener.local_addr()?);

    // Start the server and handle incoming requests
    axum::serve(listener, app).await?;

    Ok(())
}

/// The only end point that matters.
///
/// This function is an asynchronous handler that always returns the "I'm a teapot" status code.
/// HTTP 418 status code, which is defined in the Hyper Text Coffee Pot Control Protocol (HTCPCP) as a way to indicate that the server is a teapot and cannot brew coffee.
///
/// # Returns
///
/// * `StatusCode` - The "I'm a teapot" status code (HTTP 418).
///
/// # References
///
/// - [Hyper Text Coffee Pot Control Protocol (HTCPCP)](https://tools.ietf.org/html/rfc2324)
/// - [HTTP 418 Status Code](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/418)
async fn im_a_teapot() -> StatusCode {
    StatusCode::IM_A_TEAPOT
}

/// Handles the "takeoff" prediction request.
///
/// This function is an asynchronous handler for the "takeoff" prediction endpoint.
/// It adds the "takeoff" prediction to the shared state and returns a response
/// containing the prediction label and the updated count of predictions.
///
/// # Arguments
///
/// * `state` - The shared application state.
/// * `_` - The JSON value (unused in this function).
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the prediction label and count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUEST` - Returned if there is an error acquiring the lock on the predictions state.
async fn takeoff(
    state: State<AppState>,
    Json(_): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(String::from("takeoff"));

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();
    Ok(PredictionResponse {
        prediction_label: "takeoff".into(),
        prediction_count,
    }
    .into())
}

/// Handles the "land" prediction request.
///
/// This function is an asynchronous handler for the "land" prediction endpoint.
/// It adds the "land" prediction to the shared state and returns a response
/// containing the prediction label and the updated count of predictions.
///
/// # Arguments
///
/// * `state` - The shared application state.
/// * `_` - The JSON value (unused in this function).
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the prediction label and count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUEST` - Returned if there is an error acquiring the lock on the predictions state.
async fn land(
    state: State<AppState>,
    Json(_): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(String::from("land"));

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();
    Ok(PredictionResponse {
        prediction_label: "land".into(),
        prediction_count,
    }
    .into())
}

/// Handles the "forward" prediction request.
///
/// This function is an asynchronous handler for the "forward" prediction endpoint.
/// It adds the "forward" prediction to the shared state and returns a response
/// containing the prediction label and the updated count of predictions.
///
/// # Arguments
///
/// * `state` - The shared application state.
/// * `_` - The JSON value (unused in this function).
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the prediction label and count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUEST` - Returned if there is an error acquiring the lock on the predictions state.
async fn forward(
    state: State<AppState>,
    Json(_): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(String::from("forward"));

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();
    Ok(PredictionResponse {
        prediction_label: "forward".into(),
        prediction_count,
    }
    .into())
}

/// Handles the "backward" prediction request.
///
/// This function is an asynchronous handler for the "backward" prediction endpoint.
/// It adds the "backward" prediction to the shared state and returns a response
/// containing the prediction label and the updated count of predictions.
///
/// # Arguments
///
/// * `state` - The shared application state.
/// * `_` - The JSON value (unused in this function).
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the prediction label and count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUEST` - Returned if there is an error acquiring the lock on the predictions state.
async fn backward(
    state: State<AppState>,
    Json(_): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(String::from("backward"));

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();
    Ok(PredictionResponse {
        prediction_label: "backward".into(),
        prediction_count,
    }
    .into())
}

/// Handles the "right" prediction request.
///
/// This function is an asynchronous handler for the "right" prediction endpoint.
/// It adds the "right" prediction to the shared state and returns a response
/// containing the prediction label and the updated count of predictions.
///
/// # Arguments
///
/// * `state` - The shared application state.
/// * `_` - The JSON value (unused in this function).
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the prediction label and count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUEST` - Returned if there is an error acquiring the lock on the predictions state.
async fn right(
    state: State<AppState>,
    Json(_): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(String::from("right"));

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();
    Ok(PredictionResponse {
        prediction_label: "right".into(),
        prediction_count,
    }
    .into())
}

/// Handles the "left" prediction request.
///
/// This function is an asynchronous handler for the "left" prediction endpoint.
/// It adds the "left" prediction to the shared state and returns a response
/// containing the prediction label and the updated count of predictions.
///
/// # Arguments
///
/// * `state` - The shared application state.
/// * `_` - The JSON value (unused in this function).
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the prediction label and count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUEST` - Returned if there is an error acquiring the lock on the predictions state.
async fn left(
    state: State<AppState>,
    Json(_): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(String::from("left"));

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();
    Ok(PredictionResponse {
        prediction_label: "left".into(),
        prediction_count,
    }
    .into())
}

/// Handles the prediction request when the input data is stored in vectors.
///
/// This function is a handler for the prediction endpoint.
/// It expects the input data to be provided as a JSON object containing vectors.
/// The function converts the JSON data to a DataFrame, prepares the input tensor,
/// and makes a prediction using the loaded TensorFlow model. The prediction is then
/// added to the shared state, and the response is returned as a JSON value.
///
/// # Arguments
///
/// * `state` - The shared application state containing the TensorFlow session, graph, and other resources.
/// * `json` - The JSON value containing the input data for the prediction.
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the prediction label and count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUEST` - Returned if there is an error during the prediction process,
///   such as invalid input data or an error in the TensorFlow model.
async fn egg_prediction(
    state: State<AppState>,
    Json(json): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    let df = vec_json_to_dataframe(json).or(Err(StatusCode::BAD_REQUEST))?;

    let input_tensor = df_to_tensor(df).or(Err(StatusCode::BAD_REQUEST))?;

    // Make the prediction using the loaded TensorFlow model
    let prediction = predict_motion(&state.session, &state.graph, &input_tensor)
        .or(Err(StatusCode::BAD_REQUEST))?;

    println!("{:?}", prediction);

    // Add the prediction to the shared state
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(prediction.into());

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();

    Ok(PredictionResponse {
        prediction_label: prediction.into(),
        prediction_count,
    }
    .into())
}

async fn egg_timestamp_prediction(
    state: State<AppState>,
    Json(json): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    let df = vec_json_to_dataframe_timestamp(json).or(Err(StatusCode::BAD_REQUEST))?;

    let input_tensor = df_to_tensor(df).or(Err(StatusCode::BAD_REQUEST))?;

    // Make the prediction using the loaded TensorFlow model
    let prediction = predict_motion(&state.session, &state.graph, &input_tensor)
        .or(Err(StatusCode::BAD_REQUEST))?;

    println!("{:?}", prediction);

    // Add the prediction to the shared state
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(prediction.into());

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();

    Ok(PredictionResponse {
        prediction_label: prediction.into(),
        prediction_count,
    }
    .into())
}
/// Retrieves the last prediction made by the application.
///
/// This function is an asynchronous handler for the "/lastpredition" route.
/// It accesses the shared state to retrieve the last prediction and its count.
/// If no predictions have been made yet, it returns a default response with
/// a "land" prediction label and a count of 0.
///
/// # Arguments
///
/// * `state` - The shared application state containing the predictions.
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response
///   with the last prediction and its count on success, or a `StatusCode` indicating an error.
///
/// # Errors
///
/// * `StatusCode::BAD_REQUSET` - Returned if the lock on the predictions vector cannot be acquired.
async fn last_prediction(state: State<AppState>) -> Result<Json<PredictionResponse>, StatusCode> {
    let predictions = state.perdictions.lock().or(Err(StatusCode::BAD_REQUEST))?;

    let Some(prediction_label) = predictions.last() else {
        return Ok(PredictionResponse {
            prediction_label: "land".into(),
            prediction_count: 0,
        }
        .into());
    };

    Ok(PredictionResponse {
        prediction_label: prediction_label.clone(),
        prediction_count: predictions.len(),
    }
    .into())
}

/// Handles the POST request for making predictions.
///
/// This function takes a JSON value as input.
/// It converts the JSON value to a dataframe, prepares the input tensor, and
/// makes a prediction using the loaded TensorFlow model. The prediction is then
/// added to the shared state, and the response is returned as a JSON value.
///
/// # Arguments
///
/// * `state` - The shared application state containing the TensorFlow session, graph, and other resources.
/// * `val` - The JSON value containing the input data for the prediction.
///
/// # Returns
///
/// * `Result<Json<PredictionResponse>, StatusCode>` - A `Result` containing the JSON response with the prediction
///   label and count on success, or a `StatusCode` indicating the error on failure.
async fn post_predict(
    state: State<AppState>,
    Json(json): Json<Value>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    let df = json_to_dataframe(json).unwrap();
    // .or(Err(StatusCode::BAD_REQUEST))?;

    let input_tensor = df_to_tensor(df).unwrap();
    // .or(Err(StatusCode::BAD_REQUEST))?;

    // Make the prediction using the loaded TensorFlow model
    let prediction = predict_motion(&state.session, &state.graph, &input_tensor).unwrap();
    // .or(Err(StatusCode::BAD_REQUEST))?;

    println!("{:?}", prediction);

    // Add the prediction to the shared state
    state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .push(prediction.into());

    // Get the current count of predictions
    let prediction_count = state
        .perdictions
        .lock()
        .or(Err(StatusCode::BAD_REQUEST))?
        .len();

    Ok(PredictionResponse {
        prediction_label: prediction.into(),
        prediction_count,
    }
    .into())
}

/// Converts a JSON value to a Polars DataFrame.
///
/// This function takes a JSON value and converts it to a Polars DataFrame.
/// The JSON value should have the values c0-c31 as key-value pair
/// these values are objects containing the column data, of undefined size.
/// Objects store each row of a column by its index
///
/// # Arguments
///
/// * `data` - The JSON value to be converted to a DataFrame.
///
/// # Returns
///
/// * `Result<DataFrame>` - A `Result` containing the converted DataFrame on success,
///   or an `anyhow::Error` if the JSON format is invalid or a required column is missing.
fn json_to_dataframe(data: Value) -> Result<DataFrame> {
    let mut columns = Vec::new();

    let Value::Object(map) = data else {
        return Err(anyhow!("Invalid json format"));
    };

    // excludes cloumns c0 and c30
    for i in 1..32 {
        if i == 30 {
            continue;
        }

        // c0: Sample count index
        // c30: Time stamp
        // Iterate over column indices from c1 to c31 (excluding c30)
        let column_key = format!("c{}", i);
        let value = map
            .get(&column_key)
            .ok_or(anyhow!("missing column: {}", column_key))?;

        let Value::Object(arr) = value else {
            return Err(anyhow!("actual value {:?}", value));
        };

        let mut i = 0;
        let mut row = vec![];

        while let Some(val) = arr.get(&i.to_string()) {
            row.push(val.as_f64());
            i += 1;
        }
        let series = Series::new(&column_key, row);
        columns.push(series);
    }

    Ok(DataFrame::new(columns)?)
}

/// Converts a JSON value containing vector data to a Polars DataFrame.
///
/// This function takes a JSON value that represents a dataset with vector data
/// and converts it to a Polars DataFrame. The JSON value should have a specific
/// format where each column is represented by a key-value pair, with the key
/// being "c{i}" (where i is the column index 0-31) and the value being an array of
/// floating-point numbers.
///
/// # Arguments
///
/// * `json` - The JSON value containing the vector data to be converted to a DataFrame.
///
/// # Returns
///
/// * `Result<DataFrame>` - A `Result` containing the converted DataFrame on success,
///   or an `anyhow::Error` if the JSON format is invalid or a required column is missing.
fn vec_json_to_dataframe(json: Value) -> Result<DataFrame> {
    let mut columns = vec![];

    let Value::Object(map) = json else {
        return Err(anyhow!("Invalid json format"));
    };

    // c0: Sample count index
    // c30: Time stamp
    // Iterate over column indices from c1 to c31 (excluding c30)
    for i in 1..32 {
        if i == 30 {
            continue;
        }
        let column_key = format!("c{}", i);
        let value = map
            .get(&column_key)
            .ok_or(anyhow!("missing column: {}", column_key))?;

        let Value::Array(arr) = value else {
            return Err(anyhow!("actual value {:?}", value));
        };

        let mut row = vec![];

        // Convert each value in the array to f64 and add it to the row
        for val in arr {
            row.push(val.as_f64())
        }

        let series = Series::new(&column_key, row);
        columns.push(series);
    }
    Ok(DataFrame::new(columns)?)
}

fn vec_json_to_dataframe_timestamp(json: Value) -> Result<DataFrame> {
    let mut columns = vec![];

    let Value::Object(map) = json else {
        return Err(anyhow!("Invalid json format"));
    };

    // c0: Sample count index
    // c30: Time stamp
    // Iterate over column indices from c1 to c31 (excluding c30)
    for i in 1..32 {
        let column_key = format!("c{}", i);
        let value = map
            .get(&column_key)
            .ok_or(anyhow!("missing column: {}", column_key))?;

        let Value::Array(arr) = value else {
            return Err(anyhow!("actual value {:?}", value));
        };

        let mut row = vec![];

        // Convert each value in the array to f64 and add it to the row
        for val in arr {
            row.push(val.as_f64())
        }

        let series = Series::new(&column_key, row);
        columns.push(series);
    }
    Ok(DataFrame::new(columns)?)
}

/// Converts a Polars DataFrame to a TensorFlow Tensor.
///
/// This function takes a reference to a Polars DataFrame and converts it to a TensorFlow Tensor.
/// The DataFrame should contain only Float64 values. The resulting Tensor will have the same
/// shape as the DataFrame (num_rows x num_cols) and will be of type f32.
/// The brain reading the DataFrame will have a shape of (num_rows x 30)
/// The tensor accepts a shape of (None x 30)
///
/// # Arguments
///
/// * `df` - A reference to the Polars DataFrame to be converted to a Tensor.
///
/// # Returns
///
/// * `Result<Tensor<f32>>` - A `Result` containing the converted Tensor on success,
///   or an `anyhow::Error` if the DataFrame contains non-Float64 values.
fn df_to_tensor(df: DataFrame) -> Result<Tensor<f32>> {
    let steps = 200;
    let num_rows = df.height() - (df.height() % steps);
    let num_cols = df.width();

    let mut data = Vec::with_capacity(num_rows * num_cols);
    let mut i = 0;

    while let Ok(row) = df.get_row(i) {
        if i >= num_rows {
            break;
        }
        for val in row.0 {
            let AnyValue::Float64(val) = val else {
                return Err(anyhow!("how is this not a float {}", val));
            };
            data.push(val as f32);
        }
        i += 1;
    }

    Ok(
        Tensor::new(&[(num_rows / steps) as u64, steps as u64, num_cols as u64])
            .with_values(&data)
            .unwrap(),
    )
}

/// Predicts the motion based on the input tensor using a TensorFlow model.
///
/// This function takes a TensorFlow Session, Graph, input tensor, input operation name,
/// and output operation name, and predicts the motion using the loaded model.
/// The predicted motion is returned as a `Movements` enum variant.
///
/// # Arguments
///
/// * `session` - A reference to the TensorFlow Session.
/// * `graph` - A reference to the TensorFlow Graph.
/// * `input_tensor` - A reference to the input tensor.
///
/// # Returns
///
/// * `Result<Movements>` - A `Result` containing the predicted motion as a `Movements` enum variant on success,
///   or an `anyhow::Error` if an error occurs during the prediction process.
fn predict_motion(
    bundle: &SavedModelBundle,
    graph: &Graph,
    input_tensor: &Tensor<f32>,
) -> Result<Movements> {
    let mut args = SessionRunArgs::new();

    // Get the input operation from the graph
    let session = &bundle.session;

    let in_op = graph
        .operation_by_name_required(&INPUT_OPERATION_NAME)
        .unwrap();
    args.add_feed(&in_op, 0, input_tensor);

    let out_op = graph
        .operation_by_name_required(&OUTPUT_OPERATION_NAME)
        .unwrap();
    let result_token = args.request_fetch(&out_op, 0);

    session.run(&mut args)?;

    let result_tensor = args.fetch::<f32>(result_token).unwrap();

    let mut indexes = vec![];
    for row in result_tensor.chunks(6) {
        let index = row
            .iter()
            .enumerate()
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
            .map(|(i, _)| i)
            .ok_or(anyhow!("no predicted index found"))?;
        indexes.push(index);
    }
    let mut counts = std::collections::HashMap::with_capacity(6);
    indexes.into_iter().for_each(|i| {
        let _ = counts.entry(i).and_modify(|e| *e += 1).or_insert(1);
    });
    let (prediction, _) = counts
        .into_iter()
        .max()
        .ok_or(anyhow!("no valid indexes found"))?;
    Movements::try_from(prediction).or(Err(anyhow!("{:?}", prediction)))
}

/// Imports a TensorFlow SavedModel and returns the Session and Graph.
///
/// This function takes a file path to a TensorFlow SavedModel directory
/// and loads the model using the `tensorflow::SavedModelBundle` API.
/// It returns a tuple containing the loaded Session and Graph.
///
/// # Arguments
///
/// * `path` - The file path to the TensorFlow SavedModel directory.
///
/// # Returns
///
/// * `Result<(Session, Graph)>` - A `Result` containing a tuple of the loaded Session and Graph on success,
///   or an error if the model fails to load.
fn import_tensor(path: String) -> Result<(SavedModelBundle, Graph)> {
    // Get the current directory and append path to model
    let mut model_path = current_dir().unwrap();
    model_path.push(path);

    let mut graph = Graph::new();

    let session = tensorflow::SavedModelBundle::load(
        &SessionOptions::new(),
        &["serve"],
        &mut graph,
        model_path,
    )?;

    Ok((session, graph))
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn open_tensor() {
        let mut model_path = current_dir().unwrap();
        model_path.push("model.pb");

        let mut graph = Graph::new();

        let session = tensorflow::SavedModelBundle::load(
            &SessionOptions::new(),
            &["serve"],
            &mut graph,
            model_path,
        )
        .unwrap();
        println!(
            "{:?}",
            session.meta_graph_def().get_signature("serving_default")
        );
    }
}

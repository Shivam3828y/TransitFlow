from __future__ import annotations

import pandas as pd
import streamlit as st

from src.feature_engineering import prepare_prediction_dataset
from src.prediction_model import forecast_route, save_model, train_and_evaluate
from src.ui import dataset_signature, get_filtered_dataset, render_header
from src.visualization import (
    forecast_figure,
    prediction_comparison_figure,
)

data = get_filtered_dataset()
signature = dataset_signature(data)
render_header(
    "Demand Prediction",
    "Compare a lag-based baseline with a Random Forest using chronological "
    "holdout data, then recursively forecast future route demand.",
)

st.caption(
    f"Selected training data: {len(data):,} observations · "
    f"{st.session_state.get('dataset_name', 'Dataset')}"
)
if "service_date" not in data.columns and "service_datetime" not in data.columns:
    st.warning(
        "Prediction is unavailable because no service date is mapped. Return to "
        "Data Management and map a date or timestamp column."
    )
elif "passenger_count" not in data.columns:
    st.warning("Prediction requires passenger_count. Map that field in Data Management.")
else:
    try:
        _, _, engineered = prepare_prediction_dataset(data)
        usable_rows = len(engineered)
        usable_days = engineered["timestamp"].dt.normalize().nunique()
        st.caption(
            f"Leakage-safe feature rows: {usable_rows:,} across {usable_days:,} dates. "
            "The current passenger count is excluded from its own features."
        )
    except Exception as exc:
        usable_rows = 0
        st.warning(f"Feature preparation: {exc}")

    with st.form("prediction_training_form"):
        trees = st.slider(
            "Random Forest trees",
            min_value=40,
            max_value=300,
            value=160,
            step=20,
        )
        train = st.form_submit_button(
            "Train and evaluate models",
            type="primary",
            use_container_width=True,
        )

    if train:
        with st.spinner("Training and evaluating against a chronological holdout..."):
            try:
                model_result = train_and_evaluate(data, n_estimators=trees)
                st.session_state["model_result"] = model_result
                st.session_state["model_signature"] = signature
                st.session_state.pop("forecast_result", None)
            except Exception as exc:
                st.error(f"Could not train a reliable demand model: {exc}")

    result = st.session_state.get("model_result")
    if result and st.session_state.get("model_signature") != signature:
        st.session_state.pop("model_result", None)
        st.session_state.pop("model_signature", None)
        st.session_state.pop("forecast_result", None)
        result = None
        st.info(
            "The selected dataset or filters changed. Train again so evaluation "
            "and forecasts match the current selection."
        )

    if result:
        evaluation = result["evaluation"]
        st.success(
            f"Evaluated {evaluation['test_rows']:,} chronological holdout rows "
            f"after training on {evaluation['train_rows']:,} earlier rows. "
            f"Holdout begins {evaluation['split_date']}."
        )
        comparison_rows = []
        for name, metrics in (
            ("Lag-1 baseline", evaluation["baseline"]),
            ("Random Forest", evaluation["random_forest"]),
        ):
            comparison_rows.append(
                {
                    "model": name,
                    "MAE (passengers)": metrics["mae"],
                    "RMSE (passengers)": metrics["rmse"],
                    "R-squared": metrics["r2"],
                }
            )
        st.dataframe(
            pd.DataFrame(comparison_rows).round(3),
            hide_index=True,
            use_container_width=True,
        )
        st.plotly_chart(
            prediction_comparison_figure(result["comparison"]),
            use_container_width=True,
        )
        st.caption(evaluation["historical_feature_policy"])
        with st.expander("Prediction limitations and method"):
            st.write(
                "The baseline predicts from the prior observation in the same "
                "route/stop/hour series. The Random Forest uses shifted lag and "
                "seven-observation rolling features plus calendar and route/stop "
                "identifiers. A chronological split prevents future rows from "
                "being used for training. Forecasts are recursive, so later forecast "
                "days use earlier predictions rather than unseen actual counts."
            )
            if evaluation["random_forest"]["r2"] is None:
                st.write("R-squared is omitted because the holdout target is constant or too small.")

        st.download_button(
            "Download holdout predictions as CSV",
            data=result["comparison"].to_csv(index=False).encode("utf-8"),
            file_name="transitiq_holdout_predictions.csv",
            mime="text/csv",
        )

        if "route_id" not in data.columns:
            st.info("A mapped route ID is required for a route-specific future forecast.")
        else:
            routes = sorted(data["route_id"].dropna().astype(str).unique().tolist())
            if routes:
                st.markdown("#### Forecast a route")
                controls = st.columns([2, 1])
                route_id = controls[0].selectbox("Route", routes)
                horizon = controls[1].selectbox(
                    "Forecast days", [7, 14, 21, 30], index=1
                )
                if st.button("Generate route forecast", type="primary"):
                    try:
                        with st.spinner("Generating recursive route forecasts..."):
                            forecast = forecast_route(
                                data,
                                result["model"],
                                route_id,
                                periods=horizon,
                            )
                        st.session_state["forecast_result"] = {
                            "data": forecast,
                            "route_id": route_id,
                            "signature": signature,
                        }
                    except Exception as exc:
                        st.error(f"Forecast could not be generated: {exc}")

                forecast_state = st.session_state.get("forecast_result")
                if forecast_state and forecast_state["signature"] == signature:
                    forecast = forecast_state["data"]
                    st.plotly_chart(forecast_figure(forecast), use_container_width=True)
                    st.dataframe(forecast, hide_index=True, use_container_width=True)
                    st.download_button(
                        "Download route forecast as CSV",
                        data=forecast.to_csv(index=False).encode("utf-8"),
                        file_name=f"transitiq_forecast_{route_id}.csv",
                        mime="text/csv",
                    )

        st.markdown("#### Save trained model")
        st.caption(
            "Model files are not written unless you choose to save this trained model."
        )
        if st.button("Save model to the project", key="save_trained_model"):
            try:
                saved = save_model(result["model"])
                st.success(f"Trained model saved as {saved.name}.")
            except Exception as exc:
                st.error(f"Model could not be saved: {exc}")
    else:
        st.info(
            "No trained model is available for the current dataset and filters. "
            "Small or undated datasets are rejected with a specific reason rather "
            "than given a misleading forecast."
        )

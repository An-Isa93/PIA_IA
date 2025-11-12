document.addEventListener("DOMContentLoaded", function() {
    const modelSelect = document.getElementById("modelSelect");
    const dateField = document.getElementById("dateField");

    modelSelect.addEventListener("change", function() {
        if (modelSelect.value === "4") {
            dateField.disabled = false;
        } else {
            dateField.disabled= trues;
        }
    });
});

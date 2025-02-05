function toggleNegativePopup() {
    const negativePopup = document.getElementById('negative-popup');
    negativePopup.classList.toggle('hidden'); 
}

function togglePositivePopup() {
    const positivePopup = document.getElementById('positive-popup');
    positivePopup.classList.toggle('hidden'); 
}


document.querySelector(".input-form").addEventListener("submit", async (event) => {
    event.preventDefault(); 

    function validateForm() {
        const name = document.getElementById('name').value;
        const glucose = document.getElementById('glucose').value;
        const bmi = document.getElementById('bmi').value;
        const bloodPressure = document.getElementById('bloodPressure').value;
        const insulin = document.getElementById('insulin').value;
        const pregnancy = document.getElementById('pregnancy').value;
        const skinThickness = document.getElementById('skinThickness').value;
        const age = document.getElementById('age').value;

        console.log(name, glucose, bmi, bloodPressure, insulin, pregnancy, skinThickness, age); // Debugging

    
        if (!name || !glucose || !bmi || !bloodPressure || !insulin || !pregnancy || !skinThickness || !age) {
            alert("Please fill in all fields.");
            return false; 
        }
    
        return true;
    }

    if (!validateForm()) {
        return; 
    }

    const formData = new FormData(event.target);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (result.error) {
            alert(`Error: ${result.error}`);
            return;
        }

        if (result.prediction === 1) {
            togglePositivePopup();
        } else {
            toggleNegativePopup(); 
        }
    } catch (error) {
        alert(`An error occurred: ${error.message}`);
    }
});

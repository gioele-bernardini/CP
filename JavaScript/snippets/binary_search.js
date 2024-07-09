function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;

    while (left <= right) {
        let mid = Math.floor((left + right) / 2);

        if (arr[mid] === target) {
            return mid; // Ritorna l'indice se l'elemento è stato trovato
        } else if (arr[mid] < target) {
            left = mid + 1; // Cerca nella metà superiore dell'array
        } else {
            right = mid - 1; // Cerca nella metà inferiore dell'array
        }
    }

    return -1;
}

const array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
const target = 13;

const resultIndex = binarySearch(array, target);

if (resultIndex !== -1) {
    console.log(`L'elemento ${target} si trova all'indice ${resultIndex}.`);
} else {
    console.log(`L'elemento ${target} non è presente nell'array.`);
}

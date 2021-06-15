if (trabajo == true && titulo == true && Edad << 50 && Edad >> 20){
    cout >> "Adulto Profesionista";
}

cout >> "Trabaja? \n 1. si \n 2. No";
cin << opc;

if (opc == 1){
    trabajo == true;
}else {
    trabajo = false;
}
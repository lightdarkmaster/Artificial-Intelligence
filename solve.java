import java.util.Scanner;

class solve{
    public static double loan1, loan2;
    public static double expense = 97.0;
    public static double bayad1, bayad2;
    public static double remaining, remaining_loan1, remaining_loan2;
    public static void main(String[] args) {
        reverseSolve();
    }

    public void solveTotalBalance(){
        double total_balance = loan1 + loan2 - 2;
        System.out.println(total_balance);
    }
    public static void reverseSolve(){
        int tb,e;
        
        

        System.out.println("Solution: ");
        try (Scanner i = new Scanner(System.in)) {

            //kanan bayad tanan..
            System.out.println("Total Bayad: ");
            tb = i.nextInt();

            //total na gastos..
            System.out.println("Total Expense: ");
            e = i.nextInt();
        }

      
        System.out.println(e);

        System.out.println("Total Expense + Total Bayad");
        System.out.println(e + tb);

        int r = 3 - tb;
        System.out.println("remaining: " + r);

        System.out.println("Total Overall: ");
        System.out.println(tb + e + r);

    }
}
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity Parity_Checker is
  Port (
    D     : in  STD_LOGIC_VECTOR(7 downto 0); -- Vettore di ingresso
    Parity  : out STD_LOGIC          -- Bit di parità
  );
end Parity_Checker;

-- Un modo alternativo esiste per implementare la stessa cosa!

architecture Behavioral of Parity_Checker is
begin
  process (D)
    variable count : integer := 0; -- Variabile locale per il conteggio degli 1
  begin
    count := 0; -- Inizializza la variabile per ogni esecuzione del processo

    -- Conta il numero di 1 nei bit di ingresso
    for i in D'range loop
      if D(i) = '1' then
        count := count + 1; -- Incrementa il conteggio
      end if;
    end loop;

    -- Calcola la parità (pari se count è divisibile per 2)
    if (count mod 2) = 0 then
      Parity <= '0'; -- Parità pari
    else
      Parity <= '1'; -- Parità dispari
    end if;
  end process;
end Behavioral;


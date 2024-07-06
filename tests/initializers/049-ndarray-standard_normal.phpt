--TEST--
NDArray::standard_normal
--FILE--
<?php
try {
    $a = \NDArray::standard_normal([]);
} catch (\Throwable $t) {
    echo $t->getMessage();
}
echo PHP_EOL;
$a = \NDArray::standard_normal([4]);
print_r(count($a->toArray()));
foreach ($a->toArray() as $el) {
    if (is_float($el)) {
        echo PHP_EOL . 'true';
    }
}
echo PHP_EOL;
$a = \NDArray::standard_normal([4, 4]);
print_r(count($a->toArray()));
print_r(count($a->toArray()[0]));
print_r(count($a->toArray()[1]));
print_r(count($a->toArray()[2]));
print_r(count($a->toArray()[3]));
foreach ($a->toArray() as $el) {
    foreach ($el as $subEl) {
        if (is_float($subEl)) {
            echo PHP_EOL . 'true';
        }
    }
}
?>
--EXPECT--
Invalid parameter: Expected a non-empty array.
4
true
true
true
true
44444
true
true
true
true
true
true
true
true
true
true
true
true
true
true
true
true
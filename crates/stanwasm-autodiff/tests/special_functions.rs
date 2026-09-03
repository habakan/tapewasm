//! The special functions against a reference computed at 60 decimal digits.
#![allow(clippy::approx_constant)] // lgamma(3) is ln 2; the table is a table.
//!
//! The series these use are asymptotic, so their accuracy is set by where the
//! recurrence stops shifting and how many terms follow it. A gradient through
//! `student_t` is only as good as `digamma` is here.

use stanwasm_autodiff::{digamma, lgamma, trigamma};

/// Relative to the value, with an absolute floor so a near-zero root — lgamma
/// at 1 and at 2 — is not compared against nothing.
fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

const TOL: f64 = 1e-14;

const LGAMMA: &[(f64, f64)] = &[
    (0.05, 2.9688792010517306),
    (0.1, 2.252712651734206),
    (0.5, 0.5723649429247001),
    (0.9, 0.06637623973474295),
    (1.0, -1.0590312698732927e-31),
    (1.5, -0.12078223763524522),
    (2.0, -1.0590312698732927e-31),
    (2.5, 0.2846828704729192),
    (3.0, 0.6931471805599453),
    (4.0, 1.791759469228055),
    (5.5, 3.9578139676187165),
    (6.0, 4.787491742782046),
    (7.0, 6.579251212010101),
    (9.5, 11.689333420797269),
    (10.0, 12.801827480081469),
    (12.0, 17.502307845873887),
    (20.0, 39.339884187199495),
    (50.0, 144.5657439463449),
    (120.0, 453.0248962384961),
    (1000.0, 5905.220423209181),
];

const DIGAMMA: &[(f64, f64)] = &[
    (0.05, -20.497844991299868),
    (0.1, -10.423754940411076),
    (0.5, -1.9635100260214235),
    (0.9, -0.7549269499470513),
    (1.0, -0.5772156649015329),
    (1.5, 0.03648997397857652),
    (2.0, 0.42278433509846713),
    (2.5, 0.7031566406452432),
    (3.0, 0.9227843350984671),
    (4.0, 1.2561176684318005),
    (5.5, 1.6110931485817512),
    (6.0, 1.7061176684318005),
    (7.0, 1.8727843350984672),
    (9.5, 2.1977378764029494),
    (10.0, 2.251752589066721),
    (12.0, 2.442661679975812),
    (20.0, 2.970523992242149),
    (50.0, 3.901989673427892),
    (120.0, 4.783319289118529),
    (1000.0, 6.907255195648812),
];

const TRIGAMMA: &[(f64, f64)] = &[
    (0.05, 401.53235734211506),
    (0.1, 101.43329915079275),
    (0.5, 4.934802200544679),
    (0.9, 1.9225399594772035),
    (1.0, 1.6449340668482264),
    (1.5, 0.9348022005446793),
    (2.0, 0.6449340668482264),
    (2.5, 0.49035775610023485),
    (3.0, 0.39493406684822646),
    (4.0, 0.2838229557371153),
    (5.5, 0.19934238698962767),
    (6.0, 0.18132295573711532),
    (7.0, 0.15354517795933756),
    (9.5, 0.11099728846909904),
    (10.0, 0.10516633568168575),
    (12.0, 0.08690187287176838),
    (20.0, 0.05127082293520312),
    (50.0, 0.020201333226697125),
    (120.0, 0.008368152004833315),
    (1000.0, 0.0010005001666666333),
];

fn check(name: &str, f: fn(f64) -> f64, table: &[(f64, f64)]) {
    let mut worst = 0.0_f64;
    let mut at = 0.0;
    for &(x, want) in table {
        let e = rel(f(x), want);
        if e > worst {
            worst = e;
            at = x;
        }
    }
    assert!(
        worst < TOL,
        "{name}: worst relative error {worst:.2e} at x = {at}"
    );
}

#[test]
fn lgamma_matches_a_high_precision_reference() {
    check("lgamma", lgamma, LGAMMA);
}

#[test]
fn digamma_matches_a_high_precision_reference() {
    check("digamma", digamma, DIGAMMA);
}

#[test]
fn trigamma_matches_a_high_precision_reference() {
    check("trigamma", trigamma, TRIGAMMA);
}
